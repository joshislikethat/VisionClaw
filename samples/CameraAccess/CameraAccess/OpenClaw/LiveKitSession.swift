
import AVFoundation
import Foundation
import LiveKit
import SwiftUI

/// The entire voice+vision client, post-migration: join a LiveKit room, publish
/// mic and camera, subscribe to the agent's audio.
@MainActor
final class LiveKitSession: NSObject, ObservableObject {
  enum State: Equatable {
    case disconnected
    case connecting
    case connected
    case failed(String)
  }

  enum AgentStatus: Equatable {
    case none
    case waiting
    case starting
    case listening
    case thinking
    case speaking
    case left
  }

  @Published private(set) var state: State = .disconnected
  @Published private(set) var agentStatus: AgentStatus = .none
  @Published private(set) var isReconnecting = false
  private var isStopping = false

  @Published private(set) var usingGlassesSource = false

  struct Caption: Equatable {
    let text: String
    let isAgent: Bool
  }

  @Published private(set) var caption: Caption?
  private var captionClearTask: Task<Void, Never>?

  struct UICard: Equatable {
    struct Fact: Equatable {
      let label: String
      let value: String
    }

    struct Item: Equatable {
      let glyph: String?
      let title: String
      let subtitle: String?
      let trailing: String?
    }

    let uuid: String
    let type: String
    let title: String?
    let value: String?
    let body: String?
    let facts: [Fact]
    let items: [Item]
    let imageURL: String?
    let url: String?
    let fallbackText: String
  }

  @Published private(set) var card: UICard?

  func dismissCard() {
    card = nil
  }

  @Published private(set) var localVideoTrack: LocalVideoTrack?
  private var pendingGlassesTrack: LocalVideoTrack?
  @Published private(set) var previewTrack: LocalVideoTrack?

  let room = Room(roomOptions: RoomOptions(suspendLocalVideoTracksInBackground: false))

  override init() {
    super.init()
    room.add(delegate: self)
    AudioManager.shared.isSpeakerOutputPreferred = false
  }

  var isActive: Bool { state == .connected || state == .connecting }

  func refreshAgentStatus() {
    guard state == .connected else {
      agentStatus = .none
      return
    }

    guard let agent = room.remoteParticipants.values.first(where: { $0.isAgent }) else {
      if agentStatus != .left { agentStatus = .waiting }
      return
    }

    switch agent.agentState {
    case .idle, .initializing: agentStatus = .starting
    case .listening: agentStatus = .listening
    case .thinking: agentStatus = .thinking
    case .speaking: agentStatus = .speaking
    }
  }

  @Published private(set) var frozenFrame: UIImage?

  private let frameGrabber = LatestFrameGrabber()
  private var grabberTrack: LocalVideoTrack?

  private func attachGrabber(to track: LocalVideoTrack?) {
    if let old = grabberTrack {
      old.remove(videoRenderer: frameGrabber)
    }
    grabberTrack = track
    if let track {
      track.add(videoRenderer: frameGrabber)
    }
  }

  func toggleFreeze() async {
    if frozenFrame != nil {
      await unfreeze()
    } else {
      await freeze()
    }
  }

  func freeze() async {
    guard frozenFrame == nil, let image = frameGrabber.latestImage() else { return }
    frozenFrame = image

    if state == .connected {
      if let pub = room.localParticipant.localVideoTracks.first {
        try? await pub.mute()
      }
    }
  }

  func unfreeze() async {
    guard frozenFrame != nil else { return }
    frozenFrame = nil

    if state == .connected {
      if let pub = room.localParticipant.localVideoTracks.first {
        try? await pub.unmute()
      }
    }
  }

  @Published private(set) var zoomFactor: CGFloat = 1
  private var zoomAtGestureStart: CGFloat = 1

  private var activeCaptureDevice: AVCaptureDevice? {
    let track = localVideoTrack ?? previewTrack
    return (track?.capturer as? CameraCapturer)?.device
  }

  func beginZoomGesture() {
    zoomAtGestureStart = zoomFactor
  }

  func updateZoom(scale: CGFloat) {
    guard let device = activeCaptureDevice else { return }
    let ceiling = min(device.activeFormat.videoMaxZoomFactor, 8)
    let target = min(max(zoomAtGestureStart * scale, 1), ceiling)

    do {
      try device.lockForConfiguration()
      device.videoZoomFactor = target
      device.unlockForConfiguration()
      zoomFactor = target
    } catch {
      NSLog("[LiveKit] zoom failed: %@", error.localizedDescription)
    }
  }

  private func resetZoom() {
    zoomFactor = 1
    zoomAtGestureStart = 1
  }

  func startPreview() async {
    guard state == .disconnected || isFailed, previewTrack == nil else { return }

    let track: LocalVideoTrack

    if SettingsManager.shared.captureSource == .glasses {
      // FIX 1: added await
      track = await LocalVideoTrack.createBufferTrack(
        name: "glasses-preview",
        source: .camera
      )
      glassesCapturerBox.capturer = track.capturer as? BufferCapturer
    } else {
      // FIX 2: added await
      track = await LocalVideoTrack.createCameraTrack(
        options: CameraCaptureOptions(position: .back)
      )
      glassesCapturerBox.capturer = nil
    }

    do {
      try await track.start()
      previewTrack = track
      attachGrabber(to: track)
    } catch {
      NSLog("[LiveKit] preview camera unavailable: %@", error.localizedDescription)
    }
  }

  private final class GlassesCapturerBox: @unchecked Sendable {
    var capturer: BufferCapturer?
    var sawFrame = false
    var lastFrameAt: CFAbsoluteTime = 0
  }

  @Published private(set) var hasGlassesFrame = false
  @Published private(set) var glassesFrameStale = false
  private var glassesFrameMonitor: Task<Void, Never>?

  @Published private(set) var videoEstablishing = false
  private var callConnectedAt: CFAbsoluteTime = 0
  private var startGeneration = 0

  private let glassesCapturerBox = GlassesCapturerBox()

  nonisolated func pushGlassesFrame(_ pixelBuffer: CVPixelBuffer) {
    glassesCapturerBox.capturer?.capture(pixelBuffer)
    glassesCapturerBox.lastFrameAt = CFAbsoluteTimeGetCurrent()

    if !glassesCapturerBox.sawFrame {
      glassesCapturerBox.sawFrame = true

      Task { @MainActor in
        self.hasGlassesFrame = true
        self.videoEstablishing = false
        await self.publishPendingGlassesTrack()
      }
    }
  }

  private func stopPreview() async {
    if let track = previewTrack {
      previewTrack = nil
      try? await track.stop()
    }
  }

  func start() async {
    guard state == .disconnected || isFailed else { return }

    guard GeminiConfig.isAgentConfigured else {
      state = .failed("Gateway not configured. Check Settings.")
      return
    }

    startGeneration &+= 1
    let generation = startGeneration
    state = .connecting
    await stopPreview()

    usingGlassesSource = SettingsManager.shared.captureSource == .glasses
    AudioManager.shared.isSpeakerOutputPreferred = !usingGlassesSource

    do {
      let ticket = try await fetchTicket()

      guard generation == startGeneration,
            usingGlassesSource == (SettingsManager.shared.captureSource == .glasses) else {
        state = .disconnected
        await startPreview()
        return
      }

      try await room.connect(url: ticket.url, token: ticket.token)

      guard generation == startGeneration else {
        await room.disconnect()
        state = .disconnected
        await startPreview()
        return
      }

      try await room.localParticipant.setMicrophone(enabled: true)

      let diagSession = AVAudioSession.sharedInstance()
      NSLog(
        "[Audio] inputs=[%@] output=[%@]",
        (diagSession.availableInputs ?? []).map { $0.portType.rawValue }.joined(separator: ","),
        diagSession.currentRoute.outputs.map { $0.portType.rawValue }.joined(separator: ",")
      )

      if let hfp = diagSession.availableInputs?.first(where: { $0.portType == .bluetoothHFP }) {
        if diagSession.currentRoute.inputs.contains(where: { $0.portType == .bluetoothHFP }) {
          NSLog("[Audio] glasses HFP already routed; leaving the route alone")
        } else {
          try? diagSession.setPreferredInput(hfp)
          NSLog(
            "[Audio] selected glasses HFP; output now [%@]",
            diagSession.currentRoute.outputs.map { $0.portType.rawValue }.joined(separator: ",")
          )
        }
      }

      do {
        if usingGlassesSource {
          // FIX 3: added await
          let track = await LocalVideoTrack.createBufferTrack(
            name: "glasses",
            source: .camera,
            reportStatistics: true
          )

          glassesCapturerBox.sawFrame = false
          glassesCapturerBox.capturer = track.capturer as? BufferCapturer
          try await track.start()

          localVideoTrack = track
          pendingGlassesTrack = track
        } else {
          try await room.localParticipant.setCamera(
            enabled: true,
            captureOptions: CameraCaptureOptions(position: .back)
          )

          localVideoTrack = room.localParticipant.localVideoTracks
            .compactMap { $0.track as? LocalVideoTrack }
            .first
        }

        attachGrabber(to: localVideoTrack)
      } catch {
        NSLog("[LiveKit] camera unavailable, voice-only: %@", error.localizedDescription)
      }

      registerCaptionHandler()
      registerCardHandler()
      state = .connected
      resetZoom()
      refreshAgentStatus()

      if usingGlassesSource {
        videoEstablishing = true
        callConnectedAt = CFAbsoluteTimeGetCurrent()
        startGlassesFrameMonitor()
      }

      if glassesCapturerBox.sawFrame {
        await publishPendingGlassesTrack()
      }
    } catch {
      state = .failed(error.localizedDescription)
      agentStatus = .none
      await room.disconnect()
      await startPreview()
    }
  }

  private var glassesPublishInFlight = false

  private func publishPendingGlassesTrack() async {
    guard let track = pendingGlassesTrack,
          state == .connected,
          !glassesPublishInFlight else { return }

    glassesPublishInFlight = true
    defer { glassesPublishInFlight = false }

    do {
      _ = try await room.localParticipant.publish(
        videoTrack: track,
        options: VideoPublishOptions(
          encoding: VideoEncoding(maxBitrate: 3_000_000, maxFps: 24),
          simulcast: false,
          degradationPreference: .maintainResolution
        )
      )

      pendingGlassesTrack = nil
    } catch {
      NSLog("[LiveKit] glasses video publish failed, will retry: %@", error.localizedDescription)
    }
  }

  private func startGlassesFrameMonitor() {
    glassesFrameMonitor?.cancel()

    glassesFrameMonitor = Task { @MainActor [weak self] in
      var tick = 0

      while !Task.isCancelled {
        try? await Task.sleep(nanoseconds: 700_000_000)
        guard let self else { return }

        self.glassesFrameStale = self.hasGlassesFrame &&
          CFAbsoluteTimeGetCurrent() - self.glassesCapturerBox.lastFrameAt > 1.5

        tick += 1

        if tick % 3 == 0, let track = self.localVideoTrack {
          let publishing = track.dimensions.map {
            "\($0.width)x\($0.height)"
          } ?? "?x?"

          let out = track.statistics?.outboundRtpStream.first
          let sent: String

          if let out, let w = out.frameWidth, let h = out.frameHeight {
            let limit = out.qualityLimitationReason?.rawValue ?? "n/a"
            sent = "\(w)x\(h) @ \(String(format: "%.1f", out.framesPerSecond ?? 0)) fps, limit=\(limit)"
          } else {
            sent = "stats pending"
          }

          let appState: String
          switch UIApplication.shared.applicationState {
          case .active: appState = "active"
          case .inactive: appState = "inactive"
          case .background: appState = "background"
          @unknown default: appState = "unknown"
          }

          NSLog(
            "[VideoStats] app=%@ | publishing %@ | sent to server %@",
            appState,
            publishing,
            sent
          )
        }

        if self.videoEstablishing,
           CFAbsoluteTimeGetCurrent() - self.callConnectedAt > 6.0 {
          self.videoEstablishing = false
        }

        if self.state == .connected,
           self.pendingGlassesTrack != nil,
           self.glassesCapturerBox.sawFrame {
          await self.publishPendingGlassesTrack()
        }
      }
    }
  }

  func stop(failure: String? = nil) async {
    isStopping = true
    defer { isStopping = false }

    startGeneration &+= 1
    await room.disconnect()

    localVideoTrack = nil
    pendingGlassesTrack = nil
    state = failure.map { .failed($0) } ?? .disconnected
    agentStatus = .none
    isReconnecting = false

    resetZoom()
    frozenFrame = nil
    caption = nil
    captionClearTask?.cancel()
    card = nil

    glassesFrameMonitor?.cancel()
    glassesFrameMonitor = nil

    glassesCapturerBox.sawFrame = false
    glassesCapturerBox.lastFrameAt = 0

    hasGlassesFrame = false
    glassesFrameStale = false
    videoEstablishing = false

    await startPreview()
  }

  private func registerCaptionHandler() {
    Task { [weak self] in
      guard let room = self?.room else { return }

      try? await room.registerTextStreamHandler(for: "lk.transcription") {
        [weak self] reader, identity in

        let isAgent = identity.stringValue.hasPrefix("agent")
        var text = ""

        for try await chunk in reader {
          text += chunk
          await self?.showCaption(text, isAgent: isAgent)
        }
      }
    }
  }

  private func registerCardHandler() {
    Task { [weak self] in
      guard let room = self?.room else { return }

      try? await room.registerTextStreamHandler(for: "vc.ui") {
        [weak self] reader, _ in

        let json = try await reader.readAll()
        await self?.handleCardJSON(json)
      }
    }
  }

  private func handleCardJSON(_ json: String) {
    guard let data = json.data(using: .utf8),
          let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
          let uuid = dict["uuid"] as? String
    else {
      NSLog("[LiveKit] ignoring malformed card payload (%d bytes)", json.count)
      return
    }

    if (dict["dismiss"] as? Bool) == true {
      if card?.uuid == uuid {
        dismissCard()
      }
      return
    }

    guard let type = dict["type"] as? String else {
      NSLog("[LiveKit] ignoring card payload without a type (%d bytes)", json.count)
      return
    }

    let facts = ((dict["facts"] as? [[String: Any]]) ?? []).compactMap {
      f -> UICard.Fact? in

      guard let label = f["label"] as? String,
            let value = f["value"] as? String else { return nil }

      return UICard.Fact(label: label, value: value)
    }

    let items = ((dict["items"] as? [[String: Any]]) ?? []).compactMap {
      i -> UICard.Item? in

      guard let title = i["title"] as? String else { return nil }

      return UICard.Item(
        glyph: i["glyph"] as? String,
        title: title,
        subtitle: i["subtitle"] as? String,
        trailing: i["trailing"] as? String
      )
    }

    card = UICard(
      uuid: uuid,
      type: type,
      title: dict["title"] as? String,
      value: dict["value"] as? String,
      body: dict["body"] as? String,
      facts: facts,
      items: items,
      imageURL: dict["image_url"] as? String,
      url: dict["url"] as? String,
      fallbackText: (dict["fallback_text"] as? String) ?? ""
    )
  }

  private func showCaption(_ text: String, isAgent: Bool) {
    guard !text.isEmpty, SettingsManager.shared.showCaptions else { return }

    caption = Caption(text: text, isAgent: isAgent)
    captionClearTask?.cancel()

    captionClearTask = Task { [weak self] in
      try? await Task.sleep(nanoseconds: 4_000_000_000)
      if !Task.isCancelled {
        self?.caption = nil
      }
    }
  }

  private var isFailed: Bool {
    if case .failed = state {
      return true
    }
    return false
  }

  private struct Ticket: Decodable {
    let url: String
    let room: String
    let token: String
  }

  private func fetchTicket() async throws -> Ticket {
    guard let url = URL(string: "\(GeminiConfig.agentBaseURL)/livekit-token") else {
      throw URLError(.badURL)
    }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.timeoutInterval = 20
    request.setValue(
      "Bearer \(GeminiConfig.agentToken)",
      forHTTPHeaderField: "Authorization"
    )
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    var body: [String: String] = [
      "engine": SettingsManager.shared.intelligenceEngine.rawValue,
      "source": SettingsManager.shared.captureSource == .glasses ? "glasses" : "phone"
    ]

    if SettingsManager.shared.assistiveMode {
      body["profile"] = "assistive"
    }

    request.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (data, response) = try await URLSession.shared.data(for: request)

    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
      if (response as? HTTPURLResponse)?.statusCode == 401 {
        SettingsManager.shared.cloudGatewayToken = ""
        SettingsManager.shared.accountStatus = nil
      }

      let detail = OpenClawBridge.errorMessage(from: data) ?? "gateway error"

      throw NSError(
        domain: "LiveKitSession",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: detail]
      )
    }

    return try JSONDecoder().decode(Ticket.self, from: data)
  }
}

extension LiveKitSession: RoomDelegate {
  nonisolated func room(
    _ room: Room,
    participantDidConnect participant: RemoteParticipant
  ) {
    Task { @MainActor in
      self.refreshAgentStatus()
    }
  }

  nonisolated func room(
    _ room: Room,
    participantDidDisconnect participant: RemoteParticipant
  ) {
    let agentLeft = participant.isAgent

    Task { @MainActor in
      if agentLeft, self.state == .connected {
        self.agentStatus = .left
      } else {
        self.refreshAgentStatus()
      }
    }
  }

  nonisolated func room(
    _ room: Room,
    participant: Participant,
    didUpdateAttributes attributes: [String: String]
  ) {
    guard participant.isAgent else { return }

    Task { @MainActor in
      self.refreshAgentStatus()
    }
  }

  nonisolated func room(
    _ room: Room,
    didStartReconnectWithMode reconnectMode: ReconnectMode
  ) {
    Task { @MainActor in
      if self.state == .connected {
        self.isReconnecting = true
      }
    }
  }

  nonisolated func room(
    _ room: Room,
    didCompleteReconnectWithMode reconnectMode: ReconnectMode
  ) {
    Task { @MainActor in
      self.isReconnecting = false
    }
  }

  nonisolated func room(
    _ room: Room,
    didDisconnectWithError error: LiveKitError?
  ) {
    Task { @MainActor in
      guard !self.isStopping, self.state == .connected else { return }

      NSLog("[LiveKit] room dropped: %@", error?.localizedDescription ?? "no error")
      await self.stop(failure: "Connection lost")
    }
  }
}

final class LatestFrameGrabber: VideoRenderer {
  private let lock = NSLock()
  private var latestFrame: VideoFrame?

  var isAdaptiveStreamEnabled: Bool { false }
  var adaptiveStreamSize: CGSize { .zero }

  func set(size: CGSize) {}

  func render(frame: VideoFrame) {
    lock.lock()
    latestFrame = frame
    lock.unlock()
  }

  func render(
    frame: VideoFrame,
    captureDevice: AVCaptureDevice?,
    captureOptions: VideoCaptureOptions?
  ) {
    render(frame: frame)
  }

  func latestImage() -> UIImage? {
    lock.lock()
    let frame = latestFrame
    lock.unlock()

    guard let frame, let pixelBuffer = frame.toCVPixelBuffer() else {
      return nil
    }

    var ciImage = CIImage(cvPixelBuffer: pixelBuffer)

    switch frame.rotation {
    case ._90: ciImage = ciImage.oriented(.right)
    case ._180: ciImage = ciImage.oriented(.down)
    case ._270: ciImage = ciImage.oriented(.left)
    default: break
    }

    let context = CIContext()

    guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
      return nil
    }

    return UIImage(cgImage: cgImage)
  }
}
