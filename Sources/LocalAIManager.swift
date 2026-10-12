import AppKit
import Foundation
import Darwin
import QuartzCore
import ScreenCaptureKit
import CryptoKit

// Local AI Manager — menu bar app to run, find and clean up local LLMs (Ollama, MLX, Splash).
// macOS 27 Liquid Glass style: NSGlassEffectView popover, Activity-Monitor-style
// scrolling memory-pressure graph, per-model RAM rows with eject controls.
//
// Data sources:
//  - host_statistics64 + VM_SWAPUSAGE: RAM breakdown (app/wired/compressed/free/swap)
//  - kern.memorystatus_vm_pressure_level: system memory pressure (1 normal, 2 warning, else critical)
//  - Ollama /api/ps, /api/tags, /api/generate {keep_alive}: loaded models, unload/load/flush
//  - ps + <state dir>/mlx_servers.json: MLX server processes (start/stop/RSS)

/// Where the app keeps its own files. Moved once from the app's former folder.
enum AppPaths {
    static let stateDir: String = {
        let fm = FileManager.default
        let dir = NSString(string: "~/Library/Application Support/Local AI Manager").expandingTildeInPath
        let legacy = NSString(string: "~/.llmpanel").expandingTildeInPath
        if !fm.fileExists(atPath: dir) && fm.fileExists(atPath: legacy) {
            try? fm.moveItem(atPath: legacy, toPath: dir)
        }
        try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }()
    static let agentPrefix = "io.github.inselite.LocalAIManager"
    static let logPrefix = "/tmp/local-ai-manager"

    /// First existing executable among the usual install locations.
    static func firstExecutable(_ candidates: [String]) -> String? {
        candidates.map { NSString(string: $0).expandingTildeInPath }
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }
}

// MARK: - Data models

struct OllamaLoaded {
    let name: String
    let sizeGB: Double
    let contextLength: Int?
    let expiresAt: Date?
}

struct MlxServerState {
    var name: String
    var port: Int
    var binary: String
    var args: [String]
    var running: Bool = false
    var pid: Int? = nil
    var rssGB: Double = 0
    var weightsGB: Double = 0
    var keepRunning: Bool = false
    var configured: Bool = true
    var panel = false                 // the panel's own slot, rewritten when a model is picked
    /// Snapshot path passed to --model (used to tell which cached models are already served).
    var modelPath: String? { args.firstIndex(of: "--model").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
}

/// An MLX model already in the Hugging Face cache that one of the venv's servers can run.
struct MlxCachedModel {
    let repo: String
    let path: String                  // snapshot directory
    let server: String                // "mlx_lm.server" or "mlx_vlm.server"
    let sizeGB: Double
}

struct SplashServerState {
    var port: Int
    var label: String? = nil          // LaunchAgent for this server; nil = ad-hoc `splash serve`
    var agentModel: String? = nil     // model the LaunchAgent would start
    var logPath: String? = nil
    var running: Bool = false         // launcher or server.py process present
    var ready: Bool = false           // /status answered and reports ready
    var busy: Bool = false
    var pid: Int? = nil
    var model: String? = nil          // model actually being served
    var memGB: Double = 0             // engine's own accounting (incl. GPU + weights)
    var peakGB: Double = 0
    var contextTokens: Int? = nil
    var lastLog: String? = nil        // latest log line while starting (download/hash progress)
    var engine = "Splash"             // "Splish" for a server run from a Splish checkout

    var id: String { label ?? "pid:\(pid ?? 0)" }
    var configured: Bool { label != nil }
    var displayName: String { model ?? agentModel ?? engine.lowercased() }
    /// Drop the "org/" prefix for the row; the tooltip keeps the full id.
    var shortName: String { shortModelName(displayName) }
}

// MARK: Hugging Face

struct HFRepo {
    let id: String
    let downloads: Int
    let likes: Int
    let createdAt: Date?
}

struct HFQuant {
    let tag: String                   // e.g. "UD-Q4_K_M" — the ":VARIANT" Ollama and Splash take
    let sizeGB: Double                // summed over split parts
}

enum HFFormat: String { case gguf = "GGUF", mlx = "MLX" }

struct HFRepoDetail {
    let id: String
    let architecture: String?         // GGUF architecture, or MLX config model_type
    let contextTokens: Int?
    let quants: [HFQuant]             // MLX repos hold one quant: the whole repo
    var format: HFFormat = .gguf
    var vision = false                // MLX: config has a vision tower
}

/// A pull into Ollama, a model switch in Splash, or an MLX download + serve, shown as a progress line.
struct HFJob {
    enum Target: String { case ollama = "Ollama", splash = "Splash", mlx = "MLX" }
    let target: Target
    let model: String
    var status: String
    var fraction: Double?             // nil = indeterminate
    var failed = false
    var done = false
    var started: Date? = nil          // Splash/MLX: when the server was asked to start
}

/// "org/model:tag" → "model:tag" for row labels; tooltips keep the full id.
func shortModelName(_ name: String) -> String {
    name.split(separator: "/").last.map(String.init) ?? name
}

struct RamSample {
    var t: Date
    var usedFrac: Double
    var llmFrac: Double
}

private func hstack() -> NSStackView {
    let s = NSStackView()
    s.orientation = .horizontal
    return s
}

final class PanelModel {
    let totalGB: Double = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
    var appGB: Double = 0
    var wiredGB: Double = 0
    var compressedGB: Double = 0
    var freeGB: Double = 0
    var swapGB: Double = 0
    var pressureLevel: Int = 1

    var ollamaUp = false
    var loaded: [OllamaLoaded] = []
    var installed: [String] = []
    var mlxServers: [MlxServerState] = []
    var splashServers: [SplashServerState] = []
    var splashInstalled: [String] = []     // "org/repo:VARIANT" already in Splash's model store
    var mlxCached: [MlxCachedModel] = []
    var splashLanguageOnly: Set<String> = []  // Splash installs made with --language-only
    var splishInstalled: Set<String> = []     // installs in the Splish checkout's own store
    var splashStoreInstalled: Set<String> = [] // installs in Splash's store
    var library: LibrarySnapshot?

    var hfFormat: HFFormat = HFFormat(rawValue: UserDefaults.standard.string(forKey: "hfFormat") ?? "") ?? .gguf {
        didSet { UserDefaults.standard.set(hfFormat.rawValue, forKey: "hfFormat") }
    }
    var hfResults: [HFRepo] = []
    var hfDetail: HFRepoDetail?
    var hfSearching = false
    var hfError: String?
    var hfJob: HFJob?

    var history: [RamSample] = []
    let historyLimit = 150

    var usedFrac: Double {
        totalGB > 0 ? (appGB + wiredGB + compressedGB) / totalGB : 0
    }

    func recordSample() {
        history.append(RamSample(t: Date(), usedFrac: usedFrac, llmFrac: totalGB > 0 ? llmGB / totalGB : 0))
        if history.count > historyLimit {
            history.removeFirst(history.count - historyLimit)
        }
    }

    var pressureColor: NSColor {
        switch pressureLevel {
        case 1: return .systemGreen
        case 2: return .systemYellow
        default: return .systemRed
        }
    }

    var pressureText: String {
        switch pressureLevel {
        case 1: return "Normal"
        case 2: return "Warning"
        default: return "Critical"
        }
    }

    var llmGB: Double {
        loaded.reduce(0) { $0 + $1.sizeGB }
            + mlxServers.reduce(0) {
                guard $1.running else { return $0 }
                return $0 + ($1.weightsGB > 0 ? $1.weightsGB : $1.rssGB)
            }
            + splashGB
    }

    var splashGB: Double {
        splashServers.reduce(0) { $1.running ? $0 + $1.memGB : $0 }
    }
}

// MARK: - Memory sampling

enum MemSampler {
    static func sample(into m: PanelModel) {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(UInt32(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size))
        let host = mach_host_self()
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(host, HOST_VM_INFO64, $0, &count)
            }
        }
        var pageSize: vm_size_t = 16384
        host_page_size(host, &pageSize)
        let ps = Double(pageSize)
        if kr == KERN_SUCCESS {
            let gb: (UInt64) -> Double = { Double($0) * ps / 1_073_741_824.0 }
            m.appGB = gb(UInt64(stats.active_count) + UInt64(stats.inactive_count) + UInt64(stats.speculative_count))
            m.wiredGB = gb(UInt64(stats.wire_count))
            m.compressedGB = gb(UInt64(stats.compressor_page_count))
            m.freeGB = gb(UInt64(stats.free_count) + UInt64(stats.purgeable_count))
        }
        var swap = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        var mib: [Int32] = [CTL_VM, VM_SWAPUSAGE]
        if sysctl(&mib, 2, &swap, &size, nil, 0) == 0 {
            m.swapGB = Double(swap.xsu_used) / 1_073_741_824.0
        }
        var lvl: Int32 = 1
        var lvlLen = MemoryLayout<Int32>.size
        var name = Array("kern.memorystatus_vm_pressure_level".utf8CString)
        sysctlbyname(&name, &lvl, &lvlLen, nil, 0)
        m.pressureLevel = lvl <= 0 ? 1 : Int(lvl)
        m.recordSample()
    }
}

// MARK: - Graph view

final class ClosureButton: NSButton {
    var onAction: (() -> Void)?

    /// Title-only convenience. NOTE: NSButton(title:target:action:) clobbers the
    /// self-wiring done in init(frame:) — going through init(frame:) keeps
    /// target/action pointed at `fire` so onAction closures actually run.
    convenience init(title: String) {
        self.init(frame: NSRect(x: 0, y: 0, width: 40, height: 19))
        self.title = title
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        target = self
        action = #selector(fire)
    }
    required init?(coder: NSCoder) { fatalError() }
    @objc private func fire() { onAction?() }
}

final class PressureGraphView: NSView {
    weak var model: PanelModel?
    /// RAM used is plotted neutrally; pressure color lives on the hero chip only.
    static let usedColor = NSColor.secondaryLabelColor
    var lineColor: NSColor = PressureGraphView.usedColor

    // Samples arrive on the app-delegate's 2 s cadence; x positions are mapped from
    // real timestamps so the graph scrolls continuously instead of jumping per tick.
    static let sampleInterval: Double = 2.0

    private var fpsTimer: Timer?
    private static var maskCache: [Int: CGImage] = [:]

    override var isFlipped: Bool { true }

    private var occlusionObserver: NSObjectProtocol?

    // A closed popover keeps its window (just ordered out), so window != nil is not
    // enough — gate the 30 fps redraw on the window actually being on screen.
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if let occlusionObserver {
            NotificationCenter.default.removeObserver(occlusionObserver)
            self.occlusionObserver = nil
        }
        if let window {
            occlusionObserver = NotificationCenter.default.addObserver(
                forName: NSWindow.didChangeOcclusionStateNotification, object: window, queue: .main
            ) { [weak self] _ in self?.updateAnimation() }
        }
        updateAnimation()
    }

    func updateAnimation() {
        let onScreen = (window.map { $0.isVisible && $0.occlusionState.contains(.visible) } ?? false)
            && !isHiddenOrHasHiddenAncestor
        if onScreen, fpsTimer == nil {
            // Reduce Motion: redraw once per sample instead of scrolling continuously
            let interval = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion ? Self.sampleInterval : 1.0 / 30.0
            fpsTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
                self?.needsDisplay = true
            }
            needsDisplay = true
        } else if !onScreen {
            fpsTimer?.invalidate()
            fpsTimer = nil
        }
    }

    /// Horizontal alpha mask: quick fade-in at the left edge (old data scrolling
    /// out) and a soft fade-out toward the right edge where new samples enter.
    private static func edgeMask(width: Int) -> CGImage {
        if let cached = maskCache[width] { return cached }
        var data = [UInt8](repeating: 255, count: width)
        let fadeIn = max(6, width / 48)
        let fadeStart = Int(Double(width) * 0.93)
        for x in 0..<width {
            let a: Double
            if x < fadeIn {
                a = Double(x) / Double(fadeIn)
            } else if x >= fadeStart {
                let t = Double(x - fadeStart) / Double(width - fadeStart)
                a = 1.0 - 0.65 * t
            } else {
                a = 1.0
            }
            data[x] = UInt8(max(0, min(255, Int(a * 255.0))))
        }
        let ctx = CGContext(data: &data, width: width, height: 1, bitsPerComponent: 8,
                            bytesPerRow: width, space: nil,
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.alphaOnly.rawValue))
        let img = ctx!.makeImage()!
        maskCache[width] = img
        return img
    }

    private func addSmooth(to path: CGMutablePath, pts: [CGPoint]) {
        guard pts.count > 1 else { return }
        if pts.count == 2 {
            path.addLine(to: pts[1])
            return
        }
        for i in 0..<(pts.count - 1) {
            let p0 = pts[max(i - 1, 0)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[min(i + 2, pts.count - 1)]
            let c1 = CGPoint(x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6)
            let c2 = CGPoint(x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6)
            path.addCurve(to: p2, control1: c1, control2: c2)
        }
    }

    private func smoothPath(through pts: [CGPoint]) -> CGMutablePath {
        let path = CGMutablePath()
        guard let first = pts.first else { return path }
        path.move(to: first)
        addSmooth(to: path, pts: pts)
        return path
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width, h = bounds.height
        let gutter: CGFloat = 32
        let vPad: CGFloat = 7
        let rightPad: CGFloat = 6
        let plot = CGRect(x: gutter, y: vPad, width: w - gutter - rightPad, height: h - 2 * vPad)
        guard plot.width > 20, plot.height > 20 else { return }

        ctx.saveGState()
        // keep everything inside the card's 12pt corners
        ctx.addPath(CGPath(roundedRect: bounds.insetBy(dx: 0.5, dy: 0.5),
                           cornerWidth: 12, cornerHeight: 12, transform: nil))
        ctx.clip()

        // gridlines at 25 / 50 / 75 / 100 % with axis labels
        var gridYs: [(CGFloat, Double)] = []
        ctx.setLineWidth(0.5)
        NSColor.textColor.withAlphaComponent(0.08).setStroke()
        for frac in [0.25, 0.5, 0.75, 1.0] {
            let y = plot.minY + plot.height * CGFloat(1.0 - frac)
            gridYs.append((y, frac))
            ctx.move(to: CGPoint(x: plot.minX, y: y))
            ctx.addLine(to: CGPoint(x: plot.maxX, y: y))
        }
        ctx.strokePath()
        let labelAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .regular),
            .foregroundColor: NSColor.secondaryLabelColor,
        ]
        for (y, frac) in gridYs {
            (String(Int(frac * 100)) + "%" as NSString).draw(
                at: CGPoint(x: 4, y: y - 6.5), withAttributes: labelAttrs)
        }

        guard let model, model.history.count > 1 else {
            ctx.restoreGState()
            return
        }
        let step = plot.width / CGFloat(model.historyLimit - 1)
        let now = Date()

        func point(_ frac: Double, _ t: Date) -> CGPoint {
            // flipped coords: invert so frac 1.0 draws at the top;
            // x from real timestamps -> continuous leftward scroll
            let age = max(0, now.timeIntervalSince(t))
            let x = plot.maxX - CGFloat(age / Self.sampleInterval) * step
            return CGPoint(x: min(plot.maxX, max(plot.minX, x)),
                           y: plot.minY + plot.height * CGFloat(1.0 - min(1.0, max(0.0, frac))))
        }

        let usedPts = model.history.map { point($0.usedFrac, $0.t) }
        let llmPts = model.history.map { point($0.llmFrac, $0.t) }

        // area under the used line, filled with a vertical fade gradient
        let area = CGMutablePath()
        area.move(to: CGPoint(x: usedPts[0].x, y: plot.maxY))
        addSmooth(to: area, pts: usedPts)
        area.addLine(to: CGPoint(x: usedPts[usedPts.count - 1].x, y: plot.maxY))
        area.closeSubpath()

        ctx.saveGState()
        ctx.clip(to: CGRect(x: 0, y: 0, width: w, height: h), mask: Self.edgeMask(width: 512))

        ctx.saveGState()
        ctx.addPath(area)
        ctx.clip()
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let grad = CGGradient(colorsSpace: space, colors: [
            lineColor.withAlphaComponent(0.16).cgColor,
            lineColor.withAlphaComponent(0.02).cgColor,
        ] as CFArray, locations: [0, 1])
        if let grad {
            ctx.drawLinearGradient(grad, start: CGPoint(x: 0, y: 0), end: CGPoint(x: 0, y: h), options: [])
        }
        ctx.restoreGState()

        // LLM RAM share — dashed accent line
        ctx.addPath(smoothPath(through: llmPts))
        ctx.setStrokeColor(NSColor.controlAccentColor.withAlphaComponent(0.9).cgColor)
        ctx.setLineWidth(1.5)
        ctx.setLineCap(.round)
        ctx.setLineDash(phase: 0, lengths: [4, 3])
        ctx.strokePath()
        ctx.setLineDash(phase: 0, lengths: [])

        // system used line
        ctx.addPath(smoothPath(through: usedPts))
        ctx.setStrokeColor(lineColor.cgColor)
        ctx.setLineWidth(2.0)
        ctx.setLineJoin(.round)
        ctx.strokePath()

        ctx.restoreGState() // scroll mask

        // live dot at the newest sample — unmasked, gentle opacity pulse
        let last = usedPts[usedPts.count - 1]
        let phase = now.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.6) / 1.6
        let halo = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
            ? 0.5 : 0.5 - 0.5 * CGFloat(cos(2 * .pi * phase))
        ctx.setFillColor(lineColor.withAlphaComponent(0.10 + 0.18 * halo).cgColor)
        ctx.fillEllipse(in: CGRect(x: last.x - 7, y: last.y - 7, width: 14, height: 14))
        ctx.setFillColor(lineColor.cgColor)
        ctx.fillEllipse(in: CGRect(x: last.x - 3, y: last.y - 3, width: 6, height: 6))

        ctx.restoreGState() // rounded clip
    }
}

// MARK: - App delegate

@objc final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = PanelModel()
    var statusItem: NSStatusItem!
    var popover: NSPopover!
    private var escMonitor: Any?
    private var outsideClickMonitor: Any?
    weak var contentView: PanelContentView?
    private var refreshTimer: Timer?
    private var tickCount = 0

    let ollamaBase = "http://127.0.0.1:11434"
    let store = LibraryStore()
    var sessions: [String: LiveSession] = [:]
    var panelIntents: [String: Date] = [:]     // "backend|ref" → when the panel asked to load/unload it
    let launchedAt = Date()
    var libraryScanInFlight = false
    var libraryRescanQueued = false
    var lastJobLogged = ""
    let mlxConfigPath = "\(AppPaths.stateDir)/mlx_servers.json"
    let launchAgentsDir = NSString(string: "~/Library/LaunchAgents").expandingTildeInPath

    // MARK: lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        // One menu-bar item: opening the app while the login agent already runs it does nothing.
        if CommandLine.arguments.count == 1, let id = Bundle.main.bundleIdentifier,
           NSRunningApplication.runningApplications(withBundleIdentifier: id).count > 1 {
            exit(0)
        }
        _ = mlxTypes   // read once on the main thread; background scans use the result
        // Debug-only: `--dump-library` prints what the Library tab would list (read-only) and exits.
        if CommandLine.arguments.contains("--dump-library") {
            let types = mlxTypes
            let snap = LibraryScanner.scan { AppDelegate.mlxServer(types: types, modelType: $0, vision: $1) }
            let gb = { (b: Double) in String(format: "%.1f", b / 1_073_741_824) }
            for m in snap.models {
                print("\(m.name)  [\(m.instances.map(\.backend.badge).joined())]  \(gb(m.sizeBytes)) GB  key=\(m.key)")
                for i in m.instances {
                    print("    \(i.place): \(i.ref)  files=\(i.files.count)  alone-frees=\(gb(LibraryScanner.bytesFreed([i], in: snap))) GB")
                }
            }
            print("disk \(gb(snap.diskBytes)) GB · free \(gb(snap.freeBytes)) GB · stored twice \(gb(snap.duplicateBytes)) GB")
            for d in snap.duplicates {
                print("  \(d.name)  \(gb(d.bytes)) GB  ollama=\(d.ollamaNames)  hf-side=\(d.hfUsers.map(\.title))  → \(d.advice)")
            }
            exit(0)
        }
        loadMlxConfig()
        loadSplashAgents()
        rescanLibrary()

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.image = NSImage(systemSymbolName: "memorychip", accessibilityDescription: "Local AI Manager")
        statusItem.button?.toolTip = "Local AI Manager"
        statusItem.button?.target = self
        statusItem.button?.action = #selector(togglePopover)

        let content = PanelContentView(model: model, appDelegate: self)
        contentView = content

        popover = NSPopover()
        // .transient: dismisses when clicking outside the panel (telemetry
        // style). Esc also closes via the key monitor below.
        popover.behavior = .transient
        popover.contentSize = NSSize(width: 400, height: content.fittedHeight)
        let vc = NSViewController()
        vc.view = content
        popover.contentViewController = vc

        // Esc is an additional way to dismiss the transient popover.
        escMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] ev in
            if ev.keyCode == 53, let popover = self?.popover, popover.isShown {
                popover.performClose(nil)
                return nil
            }
            return ev
        }
        // Accessory menu-bar apps do not reliably receive AppKit's automatic
        // transient-popover dismissal when another app gets the click.
        outsideClickMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.leftMouseDown, .rightMouseDown]
        ) { [weak self] _ in
            guard let self, self.popover.isShown,
                  let window = self.popover.contentViewController?.view.window,
                  !window.frame.contains(NSEvent.mouseLocation) else { return }
            self.popover.performClose(nil)
        }

        refreshAll()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.refreshLight()
            if self.tickCount % 2 == 1 { self.refreshOllama(); self.refreshMlx(); self.refreshSplash() }
            self.tickCount += 1
        }

        if CommandLine.arguments.contains("--window") {
            showDebugWindow()
            // Debug-only: `--hf-search <query>` runs a Hugging Face search for screenshots.
            if let i = CommandLine.arguments.firstIndex(of: "--hf-search"), i + 1 < CommandLine.arguments.count {
                searchHF(CommandLine.arguments[i + 1], recentOnly: true)
                (NSApp.windows.first { $0.title == "Local AI Manager" }?.contentView as? PanelContentView)?
                    .showSearchQuery(CommandLine.arguments[i + 1])
            }
            captureDebugShotIfRequested()
        }
    }

    // Debug-only: `--shot <path> [delaySeconds]` renders the debug window
    // (own-window capture, no Screen Recording permission needed) and exits.
    private func captureDebugShotIfRequested() {
        guard let i = CommandLine.arguments.firstIndex(of: "--shot"),
              i + 1 < CommandLine.arguments.count else { return }
        let path = CommandLine.arguments[i + 1]
        let delay = i + 2 < CommandLine.arguments.count
            ? (Double(CommandLine.arguments[i + 2]) ?? 6.0)
            : 6.0
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.snapshotDebugWindow(to: path)
            NSApp.terminate(nil)
        }
    }

    private final class ShotBox {
        var data: Data?
    }

    private func snapshotDebugWindow(to path: String) {
        guard let win = NSApp.windows.first(where: { $0.title == "Local AI Manager" }) else { return }
        // `--hidpi`: draw the panel off-screen at 2x (README images on a 1x display). Liquid Glass
        // doesn't draw off-screen, so its contents go over a flat card of the glass's on-screen color.
        if CommandLine.arguments.contains("--hidpi"), let content = win.contentView {
            func findGlass(_ v: NSView) -> NSGlassEffectView? {
                (v as? NSGlassEffectView) ?? v.subviews.lazy.compactMap(findGlass).first
            }
            win.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            content.layoutSubtreeIfNeeded()
            guard let glass = findGlass(content),
                  let root = glass.subviews.first(where: { $0 is NSStackView }) else { return }
            func hiRep(_ size: NSSize) -> NSBitmapImageRep? {
                let rep = NSBitmapImageRep(
                    bitmapDataPlanes: nil, pixelsWide: Int(size.width * 2), pixelsHigh: Int(size.height * 2),
                    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
                rep?.size = size
                return rep
            }
            // trim the card to the content (the window can be taller than the current tab needs)
            var frame = glass.convert(root.bounds, from: root)
            if glass.isFlipped { frame.origin.y = glass.bounds.height - frame.maxY }
            let topMargin = glass.bounds.height - frame.maxY
            let shown = (root as? NSStackView)?.arrangedSubviews.filter { !$0.isHidden } ?? []
            let used = shown.map { root.isFlipped ? $0.frame.maxY : root.bounds.height - $0.frame.minY }.max() ?? frame.height
            let size = NSSize(width: glass.bounds.width, height: topMargin + used + min(frame.minY, 12))
            frame.origin.y = size.height - topMargin - frame.height
            guard let out = hiRep(size), let inner = hiRep(root.bounds.size) else { return }
            root.cacheDisplay(in: root.bounds, to: inner)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: out)
            let dark = win.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            NSColor(white: dark ? 45 / 255 : 246 / 255, alpha: 1).setFill()
            NSBezierPath(roundedRect: NSRect(origin: .zero, size: size), xRadius: 24, yRadius: 24).fill()
            inner.draw(in: frame, from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
            NSGraphicsContext.restoreGraphicsState()
            try? out.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
            return
        }
        let windowID = CGWindowID(win.windowNumber)
        let pxW = Int(win.frame.width * win.backingScaleFactor)
        let pxH = Int(win.frame.height * win.backingScaleFactor)
        let box = ShotBox()
        let sem = DispatchSemaphore(value: 0)
        Task.detached {
            defer { sem.signal() }
            do {
                let share = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
                guard let scWin = share.windows.first(where: { $0.windowID == windowID }) else { return }
                let filter = SCContentFilter(desktopIndependentWindow: scWin)
                let cfg = SCStreamConfiguration()
                cfg.scalesToFit = false
                cfg.width = pxW
                cfg.height = pxH
                cfg.showsCursor = false
                cfg.captureResolution = .best
                let cg = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: cfg)
                box.data = NSBitmapImageRep(cgImage: cg)
                    .representation(using: .png, properties: [:])
            } catch {
                FileHandle.standardError.write("SCK error: \(error)\n".data(using: .utf8)!)
            }
        }
        _ = sem.wait(timeout: .now() + 10)
        if let data = box.data {
            try? data.write(to: URL(fileURLWithPath: path))
            return
        }
        // fallback: layer render (misses some AppKit text but better than nothing)
        guard let content = win.contentView, let layer = content.layer else { return }
        content.layoutSubtreeIfNeeded()
        content.displayIfNeeded()
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: Int(content.bounds.width * 2),
            pixelsHigh: Int(content.bounds.height * 2), bitsPerSample: 8, samplesPerPixel: 4,
            hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
            bytesPerRow: 0, bitsPerPixel: 0) else { return }
        rep.size = content.bounds.size
        guard let ctx = NSGraphicsContext(bitmapImageRep: rep) else { return }
        ctx.cgContext.scaleBy(x: 2, y: 2)
        layer.render(in: ctx.cgContext)
        if let png = rep.representation(using: .png, properties: [:]) {
            try? png.write(to: URL(fileURLWithPath: path))
        }
    }

    @objc private func togglePopover() {
        guard let btn = statusItem.button else { return }
        if popover.isShown {
            popover.performClose(nil)
        } else {
            popover.show(relativeTo: btn.bounds, of: btn, preferredEdge: .minY)
        }
    }

    private func showDebugWindow() {
        // Debug instances run as a regular app so the window composites normally
        // (accessory-policy windows can be starved of layer contents off-screen).
        NSApp.setActivationPolicy(.regular)
        let win = NSWindow(
            contentRect: NSRect(x: 200, y: 200, width: 400, height: 600),
            styleMask: [.titled, .closable],
            backing: .buffered, defer: false
        )
        win.title = "Local AI Manager"
        win.isReleasedWhenClosed = false
        let content = PanelContentView(model: model, appDelegate: self)
        win.contentView = content
        // center on the built-in display so screencapture -R can composite it
        if let main = NSScreen.screens.first {
            let f = main.frame
            win.setFrameOrigin(NSPoint(
                x: f.midX - win.frame.width / 2,
                y: f.midY - win.frame.height / 2
            ))
        }
        win.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    // MARK: config

    private func loadMlxConfig() {
        if !FileManager.default.fileExists(atPath: mlxConfigPath) {
            // Hand-written servers go here; the app adds its own slot when you load an MLX model.
            let defaultCfg: [[String: Any]] = []
            if let data = try? JSONSerialization.data(withJSONObject: defaultCfg, options: .prettyPrinted) {
                try? FileManager.default.createDirectory(
                    atPath: (mlxConfigPath as NSString).deletingLastPathComponent,
                    withIntermediateDirectories: true)
                try? data.write(to: URL(fileURLWithPath: mlxConfigPath))
            }
        }
        guard
            let data = try? Data(contentsOf: URL(fileURLWithPath: mlxConfigPath)),
            let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return }
        model.mlxServers = arr.map { d -> MlxServerState in
            let args = d["args"] as? [String] ?? []
            var weightsGB: Double = 0
            if let mi = args.firstIndex(of: "--model"), mi + 1 < args.count {
                let rawDir = args[mi + 1]
                let dir = (rawDir as NSString).resolvingSymlinksInPath
                if let files = try? FileManager.default.contentsOfDirectory(atPath: dir) {
                    for f in files where f.hasSuffix(".safetensors") {
                        let p = (dir as NSString).appendingPathComponent(f)
                        let real = (p as NSString).resolvingSymlinksInPath
                        if let attr = try? FileManager.default.attributesOfItem(atPath: real),
                           let sz = attr[.size] as? NSNumber {
                            weightsGB += sz.doubleValue / 1_073_741_824.0
                        }
                    }
                }
            }
            return MlxServerState(
                name: d["name"] as? String ?? "server",
                port: d["port"] as? Int ?? 8080,
                binary: d["binary"] as? String ?? "",
                args: args,
                weightsGB: weightsGB,
                keepRunning: d["keep_running"] as? Bool ?? false,
                configured: true,
                panel: d["panel"] as? Bool ?? false
            )
        }
        loadMlxCached()
    }

    // MARK: refresh

    func refreshAll() {
        refreshLight()
        refreshOllama()
        refreshMlx()
        refreshSplash()
    }

    func refreshLight() {
        MemSampler.sample(into: model)
    }

    func refreshOllama() {
        fetchOllamaPs()
        fetchOllamaTags()
    }

    func refreshMlx() {
        let task = Process()
        task.launchPath = "/bin/sh"
        task.arguments = ["-c", "ps -axo pid=,rss=,command= | grep -E 'mlx_(v)?lm.*server' | grep -v grep"]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        var seen = Set<Int>()
        if (try? task.run()) != nil {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            task.waitUntilExit()
            let out = String(data: data, encoding: .utf8) ?? ""
            for line in out.split(separator: "\n") {
                let parts = line.trimmingCharacters(in: .whitespaces).split(separator: " ", maxSplits: 2)
                guard parts.count >= 2, let pid = Int(parts[0]), let rssKB = Int(parts[1]) else { continue }
                seen.insert(pid)
                let cmdline = parts.count > 2 ? String(parts[2]) : ""
                var matched = false
                for i in model.mlxServers.indices
                where cmdline.contains("--port \(model.mlxServers[i].port)") {
                    model.mlxServers[i].running = true
                    model.mlxServers[i].pid = pid
                    model.mlxServers[i].rssGB = Double(rssKB) / 1_048_576.0
                    matched = true
                }
                if !matched, !model.mlxServers.contains(where: { $0.pid == pid }) {
                    let port = cmdline.range(of: "--port (\\d+)", options: .regularExpression)
                        .map { Int(String(cmdline[$0].dropFirst("--port ".count))) } ?? 0
                    model.mlxServers.append(MlxServerState(
                        name: cmdline.contains("mlx_vlm") ? "mlx-vlm" : "mlx server",
                        port: port ?? 0, binary: "", args: [],
                        running: true, pid: pid,
                        rssGB: Double(rssKB) / 1_048_576.0,
                        keepRunning: false, configured: false
                    ))
                }
            }
        }
        for i in model.mlxServers.indices {
            if model.mlxServers[i].running, !seen.contains(model.mlxServers[i].pid ?? -1) {
                model.mlxServers[i].running = false
                model.mlxServers[i].pid = nil
                model.mlxServers[i].rssGB = 0
            }
        }
        pushUpdate()
    }

    private func fetchOllamaPs() {
        guard let url = URL(string: "\(ollamaBase)/api/ps") else { return }
        let req = URLRequest(url: url, timeoutInterval: 2.0)
        URLSession.shared.dataTask(with: req) { [weak self] data, resp, err in
            var models: [OllamaLoaded] = []
            var up = false
            if err == nil, let http = resp as? HTTPURLResponse, http.statusCode == 200,
               let data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let arr = obj["models"] as? [[String: Any]] {
                up = true
                for m in arr {
                    let name = m["name"] as? String ?? "?"
                    let bytes = (m["size_vram"] as? NSNumber)?.doubleValue
                        ?? (m["size"] as? NSNumber)?.doubleValue ?? 0
                    let ctx = (m["context_length"] as? NSNumber)?.intValue
                    var expires: Date? = nil
                    if let s = m["expires_at"] as? String {
                        let df = ISO8601DateFormatter()
                        df.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                        if let d = df.date(from: s) { expires = d }
                        else {
                            let df2 = ISO8601DateFormatter()
                            if let d = df2.date(from: s) { expires = d }
                        }
                    }
                    models.append(OllamaLoaded(name: name, sizeGB: bytes / 1_073_741_824.0,
                                               contextLength: ctx, expiresAt: expires))
                }
            }
            DispatchQueue.main.async {
                self?.model.ollamaUp = up
                self?.model.loaded = models
                self?.pushUpdate()
            }
        }.resume()
    }

    private func fetchOllamaTags() {
        guard let url = URL(string: "\(ollamaBase)/api/tags") else { return }
        let req = URLRequest(url: url, timeoutInterval: 2.0)
        URLSession.shared.dataTask(with: req) { [weak self] data, _, _ in
            var names: [String] = []
            if let data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let arr = obj["models"] as? [[String: Any]] {
                names = arr.compactMap { $0["name"] as? String }
            }
            DispatchQueue.main.async {
                self?.model.installed = names
                self?.pushUpdate()
            }
        }.resume()
    }

    func contentHeightChanged(_ height: CGFloat, for view: PanelContentView) {
        let maxH = (NSScreen.main?.visibleFrame.height ?? 900) - 40
        let h = min(max(height, 320), maxH)
        if view === contentView, let popover {
            popover.contentSize = NSSize(width: 400, height: h)
        } else if let win = view.window {
            win.setContentSize(NSSize(width: win.contentLayoutRect.width, height: h))
        }
    }

    func pushUpdate() {
        trackSessions()
        logJobTransitions()
        contentView?.rebuild()
    }

    // MARK: Ollama actions

    private func postGenerate(model name: String, keepAlive: Any) {
        let payload: [String: Any] = [
            "model": name, "prompt": "", "keep_alive": keepAlive,
            "stream": false, "options": ["num_predict": 1]
        ]
        guard let url = URL(string: "\(ollamaBase)/api/generate") else { return }
        var req = URLRequest(url: url, timeoutInterval: 180)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: payload)
        URLSession.shared.dataTask(with: req) { [weak self] _, _, _ in
            DispatchQueue.main.async {
                self?.refreshOllama()
            }
        }.resume()
    }

    func unloadModel(_ name: String) {
        notePanelIntent(.ollama, name)
        postGenerate(model: name, keepAlive: 0)
    }
    func loadModel(_ name: String) {
        notePanelIntent(.ollama, name)
        postGenerate(model: name, keepAlive: "30m")
    }
    func flushAll() {
        for m in model.loaded { unloadModel(m.name) }
    }

    // MARK: MLX actions

    func stopMlx(_ srv: MlxServerState) {
        guard let pid = srv.pid else { return }
        notePanelIntent(.mlx, srv.modelPath)
        let task = Process()
        task.launchPath = "/bin/kill"
        task.arguments = ["-TERM", "\(pid)"]
        try? task.run()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            self?.refreshMlx()
        }
    }

    func startMlx(_ srv: MlxServerState) {
        notePanelIntent(.mlx, srv.modelPath)
        _ = launchMlx(srv)
    }

    static func mlxLogPath(port: Int) -> String { "\(AppPaths.logPrefix)-mlx-\(port).log" }

    @discardableResult
    private func launchMlx(_ srv: MlxServerState) -> Process? {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: srv.binary)
        task.arguments = srv.args
        let logPath = AppDelegate.mlxLogPath(port: srv.port)
        FileManager.default.createFile(atPath: logPath, contents: nil)
        let log = FileHandle(forWritingAtPath: logPath)
        task.standardOutput = log ?? FileHandle.nullDevice
        task.standardError = log ?? FileHandle.nullDevice
        guard (try? task.run()) != nil else { return nil }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            self?.refreshMlx()
        }
        return task
    }

    // MARK: Splash

    static let splashCLI = AppPaths.firstExecutable(["~/.local/bin/splash", "/opt/homebrew/bin/splash", "/usr/local/bin/splash"])
        ?? NSString(string: "~/.local/bin/splash").expandingTildeInPath
    static let splashModelsDir = NSString(string: "~/Library/Application Support/Splash/models").expandingTildeInPath

    /// A Splish checkout (github.com/publicExcess/splish, a Splash fork tuned for M5 GPUs): the
    /// `splishDir` setting, else a checkout seen in a LaunchAgent or process, else a usual clone
    /// location. Run from source, Splish keeps its installs in <checkout>/install/models.
    static var splishRoot: String? = findSplishRoot(hint: nil)

    static func findSplishRoot(hint: String?) -> String? {
        let fm = FileManager.default
        var candidates: [String] = []
        if let dir = UserDefaults.standard.string(forKey: "splishDir"), !dir.isEmpty { candidates.append(dir) }
        if let hint { candidates.append(hint) }
        candidates += ["~/splish", "~/src/splish", "~/Developer/splish", "~/code/splish",
                       "~/projects/splish", "~/git/splish", "~/GitHub/splish"]
        // resolved, so it compares equal to the resolved install and assembly paths
        return candidates.map { (NSString(string: $0).expandingTildeInPath as NSString).resolvingSymlinksInPath }
            .first { fm.isExecutableFile(atPath: "\($0)/splish") && fm.fileExists(atPath: "\($0)/install/launcher.py") }
    }

    /// Model stores to scan: Splash's, then the Splish checkout's.
    static var splashModelRoots: [String] {
        [splashModelsDir] + (splishRoot.map { ["\($0)/install/models"] } ?? [])
    }

    /// "Splish" for paths inside the Splish checkout, else "Splash".
    static func splashEngine(for path: String) -> String {
        if let root = splishRoot, path.hasPrefix(root + "/") { return "Splish" }
        return "Splash"
    }

    /// The engine new Splash models start with: the `splashEngine` setting when Splish is present,
    /// else whichever is installed (Splash first).
    var preferredSplashEngine: String {
        guard AppDelegate.splishRoot != nil else { return "Splash" }
        if let e = UserDefaults.standard.string(forKey: "splashEngine") { return e == "Splish" ? "Splish" : "Splash" }
        return FileManager.default.isExecutableFile(atPath: AppDelegate.splashCLI) ? "Splash" : "Splish"
    }

    /// The engine that serves `id`: the one whose store has it, else the preferred one.
    func splashEngine(forModel id: String) -> String {
        guard AppDelegate.splishRoot != nil else { return "Splash" }
        let inSplish = model.splishInstalled.contains(id), inSplash = model.splashStoreInstalled.contains(id)
        if inSplish != inSplash { return inSplish ? "Splish" : "Splash" }
        return preferredSplashEngine
    }
    /// The panel's own Splash slot. Its plist is rewritten whenever a model is picked in
    /// the panel; it never starts at login and launchd never restarts it.
    static let panelSplashLabel = "\(AppPaths.agentPrefix).splash"
    static let panelSplashPort = 8000
    static let panelSplashLog = "\(AppPaths.logPrefix)-splash.log"

    private typealias SplashProc = (pid: Int, ppid: Int, port: Int, model: String?, serving: Bool, engine: String, root: String)

    /// LaunchAgents whose program is the `splash` CLI with `serve` become start-able rows.
    private func loadSplashAgents() {
        AppDelegate.splishRoot = AppDelegate.findSplishRoot(hint: AppDelegate.splishRoot)
        let files = (try? FileManager.default.contentsOfDirectory(atPath: launchAgentsDir)) ?? []
        var agents: [SplashServerState] = []
        for f in files.sorted() where f.hasSuffix(".plist") {
            let path = (launchAgentsDir as NSString).appendingPathComponent(f)
            guard
                let data = FileManager.default.contents(atPath: path),
                let plist = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
                let label = plist["Label"] as? String,
                let args = plist["ProgramArguments"] as? [String],
                let exe = args.first, ["splash", "splish"].contains((exe as NSString).lastPathComponent),
                args.contains("serve")
            else { continue }
            let isSplish = (exe as NSString).lastPathComponent == "splish"
            if isSplish, AppDelegate.splishRoot == nil {
                AppDelegate.splishRoot = AppDelegate.findSplishRoot(hint: (exe as NSString).deletingLastPathComponent)
            }
            func arg(_ flag: String) -> String? {
                args.firstIndex(of: flag).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }
            }
            agents.append(SplashServerState(
                port: arg("--port").flatMap(Int.init) ?? 8000, label: label,
                agentModel: arg("--model"),
                logPath: plist["StandardOutPath"] as? String,
                engine: isSplish ? "Splish" : "Splash"))
        }
        // keep live state of rows that survive the reload
        let previous = model.splashServers
        model.splashServers = agents.map { a in
            guard var old = previous.first(where: { $0.label == a.label }) else { return a }
            if old.agentModel != a.agentModel { old.model = nil; old.ready = false }
            old.agentModel = a.agentModel
            old.port = a.port
            old.logPath = a.logPath
            old.engine = a.engine
            return old
        } + previous.filter { !$0.configured }
        loadSplashInstalled()
    }

    /// Every install Splash links: models/<org>/<repo:VARIANT> and .selections/<hash>
    /// (installs made with options such as --language-only live only there).
    func loadSplashInstalled() {
        var names = Set<String>()
        var languageOnly = Set<String>()
        var splish = Set<String>(), splash = Set<String>()
        var seen = Set<String>()
        for link in LibraryScanner.splashLinks() {
            let assembly = (link as NSString).resolvingSymlinksInPath
            guard seen.insert(assembly).inserted,
                  let record = LibraryScanner.readJSON("\(assembly)/model.json"),
                  let id = record["model"] as? String else { continue }
            names.insert(id)
            if AppDelegate.splashEngine(for: assembly) == "Splish" { splish.insert(id) } else { splash.insert(id) }
            if record["vision_format"] as? String == "none" { languageOnly.insert(id) }
        }
        model.splashInstalled = names.sorted()
        model.splishInstalled = splish
        model.splashStoreInstalled = splash
        model.splashLanguageOnly = languageOnly
    }

    func refreshSplash() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let procs = AppDelegate.findSplashProcesses()
            let agentPids = AppDelegate.launchdPids()
            DispatchQueue.main.async {
                guard let self else { return }
                self.applySplashProcesses(procs, agentPids: agentPids)
                for srv in self.model.splashServers where srv.running {
                    self.fetchSplashStatus(id: srv.id, port: srv.port)
                }
                self.readSplashLogs()
                self.updateSplashJob()
                self.pushUpdate()
            }
        }
    }

    /// `splash serve` runs install/launcher.py (download/hash), which then becomes
    /// server/server.py; the engine is a child of server.py. A Splish checkout runs the same
    /// two scripts from its own folder.
    private static func findSplashProcesses() -> [SplashProc] {
        let out = shell("ps -axo pid=,ppid=,command= | grep -E '/(server/server\\.py|install/launcher\\.py serve)' | grep -v grep")
        var procs: [SplashProc] = []
        for line in out.split(separator: "\n") {
            let parts = line.trimmingCharacters(in: .whitespaces).split(separator: " ", maxSplits: 2,
                                                                       omittingEmptySubsequences: true)
            guard parts.count == 3, let pid = Int(parts[0]), let ppid = Int(parts[1]) else { continue }
            let cmd = String(parts[2])
            func capture(_ pattern: String) -> String? {
                guard let r = cmd.range(of: pattern, options: .regularExpression) else { return nil }
                return cmd[r].split(separator: " ").last.map(String.init)
            }
            // the checkout the scripts run from; only Splash-shaped folders count
            guard let script = cmd.split(separator: " ").first(where: {
                $0.hasSuffix("/server/server.py") || $0.hasSuffix("/install/launcher.py")
            }) else { continue }
            let root = ((String(script) as NSString).deletingLastPathComponent as NSString).deletingLastPathComponent
            guard FileManager.default.fileExists(atPath: "\(root)/install/launcher.py") else { continue }
            let splish = FileManager.default.isExecutableFile(atPath: "\(root)/splish")
            procs.append((pid, ppid, capture("--port [0-9]+").flatMap(Int.init) ?? 8000,
                          capture("--model [^ ]+"), cmd.contains("server/server.py"),
                          splish ? "Splish" : "Splash", root))
        }
        // a launcher that forked its server would show up twice — keep the server
        return procs.filter { p in !procs.contains { $0.ppid == p.pid && $0.serving } }
    }

    /// label → pid for every loaded launchd job that is currently running.
    private static func launchdPids() -> [String: Int] {
        var pids: [String: Int] = [:]
        for line in shell("/bin/launchctl list").split(separator: "\n") {
            let cols = line.split(separator: "\t")
            if cols.count == 3, let pid = Int(cols[0]) { pids[String(cols[2])] = pid }
        }
        return pids
    }

    private static func shell(_ command: String) -> String {
        let task = Process()
        task.launchPath = "/bin/sh"
        task.arguments = ["-c", command]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        guard (try? task.run()) != nil else { return "" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        return String(data: data, encoding: .utf8) ?? ""
    }

    private func applySplashProcesses(_ procs: [SplashProc], agentPids: [String: Int]) {
        var claimed = Set<Int>()
        func markRunning(_ i: Int, pid: Int, proc: SplashProc?) {
            if model.splashServers[i].pid != pid { model.splashServers[i].ready = false }
            model.splashServers[i].running = true
            model.splashServers[i].pid = pid
            if let proc {
                model.splashServers[i].port = proc.port
                model.splashServers[i].engine = proc.engine
                if proc.engine == "Splish", AppDelegate.splishRoot == nil {
                    AppDelegate.splishRoot = AppDelegate.findSplishRoot(hint: proc.root)
                    loadSplashInstalled()
                }
            }
            if model.splashServers[i].model == nil { model.splashServers[i].model = proc?.model }
        }
        // agents: launchd knows their pid; the server may be that pid or its child
        for i in model.splashServers.indices where model.splashServers[i].configured {
            let s = model.splashServers[i]
            if let pid = agentPids[s.label!] {
                let proc = procs.first { $0.pid == pid } ?? procs.first { $0.ppid == pid }
                markRunning(i, pid: proc?.pid ?? pid, proc: proc)
                if let proc { claimed.insert(proc.pid) }
            } else {
                model.splashServers[i] = SplashServerState(port: s.port, label: s.label,
                                                           agentModel: s.agentModel, logPath: s.logPath, engine: s.engine)
            }
        }
        // everything else is an ad-hoc `splash serve`
        let livePids = Set(procs.map(\.pid))
        model.splashServers.removeAll { !$0.configured && !livePids.contains($0.pid ?? -1) }
        for p in procs where !claimed.contains(p.pid) {
            if let i = model.splashServers.firstIndex(where: { !$0.configured && $0.pid == p.pid }) {
                markRunning(i, pid: p.pid, proc: p)
            } else {
                model.splashServers.append(SplashServerState(port: p.port, running: true, pid: p.pid, model: p.model,
                                                             engine: p.engine))
            }
        }
        // running first, then the panel slot, then other agents by label
        model.splashServers.sort { a, b in
            if a.running != b.running { return a.running }
            let ap = a.label == AppDelegate.panelSplashLabel, bp = b.label == AppDelegate.panelSplashLabel
            if ap != bp { return ap }
            return a.id < b.id
        }
    }

    private func fetchSplashStatus(id: String, port: Int) {
        guard let url = URL(string: "http://127.0.0.1:\(port)/status") else { return }
        URLSession.shared.dataTask(with: URLRequest(url: url, timeoutInterval: 2.0)) { [weak self] data, resp, _ in
            var obj: [String: Any]?
            if let http = resp as? HTTPURLResponse, http.statusCode == 200, let data {
                obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            }
            DispatchQueue.main.async {
                guard let self,
                      let i = self.model.splashServers.firstIndex(where: { $0.id == id && $0.running })
                else { return }
                guard let obj else {
                    self.model.splashServers[i].ready = false
                    self.pushUpdate()
                    return
                }
                func num(_ path: String...) -> Double? {
                    var cur: Any? = obj
                    for k in path { cur = (cur as? [String: Any])?[k] }
                    return (cur as? NSNumber)?.doubleValue
                }
                let gib = 1_073_741_824.0
                var s = self.model.splashServers[i]
                s.ready = obj["ready"] as? Bool ?? false
                if let cur = num("memory_actual", "current_bytes") ?? num("memory_governor", "observed_resident_bytes") {
                    s.memGB = cur / gib
                }
                s.peakGB = (num("memory_actual", "peak_bytes") ?? 0) / gib
                s.contextTokens = num("maximum_context_tokens").map { Int($0) }
                let active = ["queued", "prefilling", "decoding"].reduce(0.0) { $0 + (num("scheduler", $1) ?? 0) }
                s.busy = active > 0
                if let m = (obj["instance"] as? [String: Any])?["model"] as? String { s.model = m }
                if s.ready { s.lastLog = nil }
                self.model.splashServers[i] = s
                self.updateSplashJob()
                self.pushUpdate()
            }
        }.resume()
    }

    /// While a server is starting (download, hashing, warm-up) its log is the only progress signal.
    private func readSplashLogs() {
        for i in model.splashServers.indices {
            let s = model.splashServers[i]
            let wantsLog = (s.running && !s.ready)
                || (s.label == AppDelegate.panelSplashLabel && model.hfJob?.target == .splash)
            guard wantsLog, let path = s.logPath else { continue }
            model.splashServers[i].lastLog = AppDelegate.lastLogLine(path)
        }
    }

    static func lastLogLine(_ path: String) -> String? {
        guard let fh = FileHandle(forReadingAtPath: path) else { return nil }
        defer { try? fh.close() }
        let end = (try? fh.seekToEnd()) ?? 0
        try? fh.seek(toOffset: end > 4096 ? end - 4096 : 0)
        guard let data = try? fh.readToEnd(), let text = String(data: data, encoding: .utf8) else { return nil }
        let line = text.components(separatedBy: CharacterSet(charactersIn: "\r\n"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .last { !$0.isEmpty }
        // drop the "HH:MM:SS " stamp Splash puts on each line
        return line.map { $0.replacingOccurrences(of: "^[0-9]{2}:[0-9]{2}:[0-9]{2} ", with: "",
                                                   options: .regularExpression) }
    }

    func stopSplash(_ srv: SplashServerState) {
        guard let pid = srv.pid else { return }
        notePanelIntent(.splash, srv.model ?? srv.agentModel)
        kill(pid_t(pid), SIGTERM)  // server.py shuts the engine down cleanly on TERM
        for delay in [1.5, 5.0] {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in self?.refreshSplash() }
        }
    }

    /// Starts (or switches to) an agent via launchd, so the panel and the agents never race
    /// for a port: whatever else serves that port is stopped first.
    func startSplash(_ srv: SplashServerState) {
        guard let label = srv.label else { return }
        notePanelIntent(.splash, srv.agentModel)
        for s in model.splashServers where s.running && s.port == srv.port { notePanelIntent(.splash, s.model) }
        freeSplashPort(srv.port, except: srv.id) { [weak self] in
            self?.launchAgent(label)
        }
    }

    /// Point the panel's Splash slot at `modelID` ("org/repo:VARIANT") and start it.
    /// Splash downloads and verifies the model itself on first use.
    func serveInSplash(_ modelID: String, languageOnly: Bool = false) {
        let engine = splashEngine(forModel: modelID)
        notePanelIntent(.splash, modelID)
        for s in model.splashServers where s.running && s.port == AppDelegate.panelSplashPort { notePanelIntent(.splash, s.model) }
        model.hfJob = HFJob(target: .splash, model: modelID, status: "Stopping the server on :\(AppDelegate.panelSplashPort)…")
        pushUpdate()
        freeSplashPort(AppDelegate.panelSplashPort, except: nil) { [weak self] in
            guard let self else { return }
            self.writePanelSplashAgent(model: modelID, languageOnly: languageOnly, engine: engine)
            let domain = "gui/\(getuid())"
            _ = AppDelegate.launchctl(["bootout", "\(domain)/\(AppDelegate.panelSplashLabel)"])
            _ = AppDelegate.launchctl(["bootstrap", domain, self.panelSplashPlistPath])
            self.loadSplashAgents()
            self.launchAgent(AppDelegate.panelSplashLabel)
            self.model.hfJob?.status = "Starting \(engine)…"
            self.model.hfJob?.started = Date()
            self.pushUpdate()
        }
    }

    private var panelSplashPlistPath: String {
        (launchAgentsDir as NSString).appendingPathComponent("\(AppDelegate.panelSplashLabel).plist")
    }

    private func writePanelSplashAgent(model modelID: String, languageOnly: Bool, engine: String) {
        let home = NSHomeDirectory()
        let cli = engine == "Splish" ? AppDelegate.splishRoot.map { "\($0)/splish" } ?? AppDelegate.splashCLI
                                     : AppDelegate.splashCLI
        let plist: [String: Any] = [
            "Label": AppDelegate.panelSplashLabel,
            "ProgramArguments": [cli, "serve", "--model", modelID]
                + (languageOnly ? ["--language-only"] : [])
                + ["--host", "127.0.0.1", "--port", "\(AppDelegate.panelSplashPort)"],
            "RunAtLoad": false,
            "KeepAlive": false,
            "EnvironmentVariables": ["PATH": "\(home)/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"],
            "StandardOutPath": AppDelegate.panelSplashLog,
            "StandardErrorPath": AppDelegate.panelSplashLog,
        ]
        // fresh log per model, so progress and errors belong to this run
        FileManager.default.createFile(atPath: AppDelegate.panelSplashLog, contents: Data())
        if let data = try? PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0) {
            try? data.write(to: URL(fileURLWithPath: panelSplashPlistPath))
        }
    }

    /// SIGTERM every Splash server on `port` (except `id`), wait up to 60 s for them to exit, then continue.
    private func freeSplashPort(_ port: Int, except id: String?, then next: @escaping () -> Void) {
        let pids = model.splashServers
            .filter { $0.running && $0.port == port && $0.id != id }
            .compactMap(\.pid)
        guard !pids.isEmpty else { next(); return }
        pids.forEach { kill(pid_t($0), SIGTERM) }
        DispatchQueue.global(qos: .utility).async {
            let deadline = Date().addingTimeInterval(60)
            while Date() < deadline, pids.contains(where: { kill(pid_t($0), 0) == 0 }) {
                usleep(250_000)
            }
            DispatchQueue.main.async(execute: next)
        }
    }

    private func launchAgent(_ label: String) {
        let domain = "gui/\(getuid())"
        if AppDelegate.launchctl(["kickstart", "\(domain)/\(label)"]) != 0 {
            // not bootstrapped (e.g. after a bootout) — load it, then start it
            let plist = (launchAgentsDir as NSString).appendingPathComponent("\(label).plist")
            _ = AppDelegate.launchctl(["bootstrap", domain, plist])
            _ = AppDelegate.launchctl(["kickstart", "\(domain)/\(label)"])
        }
        for delay in [1.0, 3.0] {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in self?.refreshSplash() }
        }
    }

    @discardableResult
    private static func launchctl(_ args: [String]) -> Int32 {
        let task = Process()
        task.launchPath = "/bin/launchctl"
        task.arguments = args
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice
        guard (try? task.run()) != nil else { return -1 }
        task.waitUntilExit()
        return task.terminationStatus
    }

    /// Follows a panel-started Splash model through download → warm-up → ready (or failure).
    private func updateSplashJob() {
        guard var job = model.hfJob, job.target == .splash, !job.done, !job.failed,
              let started = job.started else { return }
        let slot = model.splashServers.first { $0.label == AppDelegate.panelSplashLabel }
        if let slot, slot.running, slot.ready {
            job.done = true
            job.fraction = 1
            job.status = "Ready on :\(slot.port)"
        } else if let slot, slot.running {
            job.status = slot.lastLog ?? "Starting Splash…"
        } else if Date().timeIntervalSince(started) > 8 {
            job.failed = true
            job.status = slot?.lastLog ?? "Splash exited — see \(AppDelegate.panelSplashLog)"
        }
        model.hfJob = job
    }

    // MARK: Hugging Face

    private let hfBase = "https://huggingface.co/api"
    private var hfSearchTask: URLSessionDataTask?
    private var hfDetailID: String?
    private var pullTask: Task<Void, Never>?

    /// Read once from the huggingface_hub token file, so gated repos list and resolve.
    private lazy var hfToken: String? = {
        let path = NSString(string: "~/.cache/huggingface/token").expandingTildeInPath
        return (try? String(contentsOfFile: path, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }()

    private func hfRequest(_ path: String, query: [URLQueryItem] = []) -> URLRequest? {
        guard var comps = URLComponents(string: hfBase + path) else { return nil }
        if !query.isEmpty { comps.queryItems = query }
        guard let url = comps.url else { return nil }
        var req = URLRequest(url: url, timeoutInterval: 15)
        if let hfToken, !hfToken.isEmpty { req.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization") }
        return req
    }

    /// Text models whose pipeline tag (when set) means a chat/completions server can use them.
    static let textPipelines: Set<String> = ["text-generation", "image-text-to-text", "any-to-any", "text2text-generation"]

    /// GGUF or MLX repos by downloads; `recentOnly` keeps those created in the last 90 days.
    func searchHF(_ query: String, recentOnly: Bool) {
        hfSearchTask?.cancel()
        let format = model.hfFormat
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty, let req = hfRequest("/models", query: [
            URLQueryItem(name: "search", value: q),
            URLQueryItem(name: "filter", value: format == .gguf ? "gguf" : "mlx"),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "60"),
            URLQueryItem(name: "expand[]", value: "downloads"),
            URLQueryItem(name: "expand[]", value: "likes"),
            URLQueryItem(name: "expand[]", value: "createdAt"),
            URLQueryItem(name: "expand[]", value: "pipeline_tag"),
        ]) else { return }
        model.hfSearching = true
        model.hfError = nil
        pushUpdate()
        let task = URLSession.shared.dataTask(with: req) { [weak self] data, resp, err in
            if (err as? URLError)?.code == .cancelled { return }
            let iso = ISO8601DateFormatter()
            iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            var repos: [HFRepo] = []
            var error: String?
            if let arr = data.flatMap({ try? JSONSerialization.jsonObject(with: $0) as? [[String: Any]] }) {
                repos = arr.compactMap { d in
                    guard let id = d["id"] as? String else { return nil }
                    if let tag = d["pipeline_tag"] as? String, !AppDelegate.textPipelines.contains(tag) { return nil }
                    return HFRepo(id: id, downloads: d["downloads"] as? Int ?? 0, likes: d["likes"] as? Int ?? 0,
                                  createdAt: (d["createdAt"] as? String).flatMap(iso.date(from:)))
                }
            } else {
                error = err?.localizedDescription ?? "Hugging Face search failed (HTTP \((resp as? HTTPURLResponse)?.statusCode ?? 0))"
            }
            if recentOnly {
                let cutoff = Date().addingTimeInterval(-90 * 86_400)
                repos = repos.filter { ($0.createdAt ?? .distantPast) >= cutoff }
            }
            repos = Array(repos.prefix(25))
            DispatchQueue.main.async {
                guard let self, self.model.hfFormat == format else { return }
                self.model.hfSearching = false
                self.model.hfResults = repos
                let kind = format.rawValue
                self.model.hfError = error ?? (repos.isEmpty ? (recentOnly ? "No \(kind) text models from the last 90 days" : "No \(kind) text models found") : nil)
                self.model.hfDetail = nil
                self.pushUpdate()
                if let first = repos.first { self.fetchHFDetail(first.id) }
            }
        }
        hfSearchTask = task
        task.resume()
    }

    /// Architecture/context from the repo's GGUF metadata, quants from its file tree.
    func fetchHFDetail(_ id: String) {
        hfDetailID = id
        model.hfDetail = nil
        pushUpdate()
        if model.hfFormat == .mlx { fetchMlxDetail(id); return }
        guard let infoReq = hfRequest("/models/\(id)", query: [URLQueryItem(name: "expand[]", value: "gguf")]),
              let treeReq = hfRequest("/models/\(id)/tree/main", query: [URLQueryItem(name: "recursive", value: "true")])
        else { return }
        let group = DispatchGroup()
        var info: [String: Any]?
        var files: [[String: Any]] = []
        group.enter()
        URLSession.shared.dataTask(with: infoReq) { data, _, _ in
            info = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            group.leave()
        }.resume()
        group.enter()
        URLSession.shared.dataTask(with: treeReq) { data, _, _ in
            files = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [[String: Any]] } ?? []
            group.leave()
        }.resume()
        group.notify(queue: .main) { [weak self] in
            guard let self, self.hfDetailID == id else { return }
            let gguf = info?["gguf"] as? [String: Any]
            var arch = gguf?["architecture"] as? String
            if arch == "clip" { arch = nil }  // metadata came from an mmproj file
            self.model.hfDetail = HFRepoDetail(
                id: id, architecture: arch,
                contextTokens: (gguf?["context_length"] as? NSNumber)?.intValue,
                quants: AppDelegate.quants(from: files))
            if self.model.hfDetail?.quants.isEmpty == true {
                self.model.hfError = "No GGUF weight files in \(id)"
            } else {
                self.model.hfError = nil
            }
            self.pushUpdate()
        }
    }

    /// Group weight files by quant tag (split parts summed); skip projectors, imatrix, MTP heads.
    static func quants(from files: [[String: Any]]) -> [HFQuant] {
        var sizes: [String: Double] = [:]
        for f in files {
            guard f["type"] as? String == "file", let path = f["path"] as? String,
                  let tag = LibraryScanner.ggufQuantTag(path) else { continue }
            let size = ((f["lfs"] as? [String: Any])?["size"] as? NSNumber ?? f["size"] as? NSNumber)?.doubleValue ?? 0
            sizes[tag, default: 0] += size / 1_073_741_824.0
        }
        return sizes.map { HFQuant(tag: $0.key, sizeGB: $0.value) }.sorted { $0.sizeGB < $1.sizeGB }
    }

    /// MLX repos: model type, context and quantization from config.json; size from the safetensors.
    private func fetchMlxDetail(_ id: String) {
        guard let treeReq = hfRequest("/models/\(id)/tree/main", query: [URLQueryItem(name: "recursive", value: "true")]),
              let cfgURL = URL(string: "https://huggingface.co/\(id)/resolve/main/config.json")
        else { return }
        var cfgReq = URLRequest(url: cfgURL, timeoutInterval: 15)
        if let hfToken, !hfToken.isEmpty { cfgReq.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization") }
        let group = DispatchGroup()
        var cfg: [String: Any]?
        var files: [[String: Any]] = []
        group.enter()
        URLSession.shared.dataTask(with: cfgReq) { data, _, _ in
            cfg = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            group.leave()
        }.resume()
        group.enter()
        URLSession.shared.dataTask(with: treeReq) { data, _, _ in
            files = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [[String: Any]] } ?? []
            group.leave()
        }.resume()
        group.notify(queue: .main) { [weak self] in
            guard let self, self.hfDetailID == id, self.model.hfFormat == .mlx else { return }
            let text = cfg?["text_config"] as? [String: Any]
            let ctx = (text?["max_position_embeddings"] ?? cfg?["max_position_embeddings"]) as? NSNumber
            let quant = (cfg?["quantization"] ?? cfg?["quantization_config"]) as? [String: Any]
            var tag = "BF16"
            if let mode = quant?["mode"] as? String, mode != "affine" {
                tag = mode.uppercased()                      // mxfp4 / mxfp8 / nvfp4
            } else if let bits = (quant?["bits"] as? NSNumber)?.intValue {
                tag = "\(bits)-bit"
            } else if let dtype = (cfg?["torch_dtype"] ?? text?["torch_dtype"] ?? cfg?["dtype"]) as? String {
                tag = ["bfloat16": "BF16", "float16": "F16", "float32": "F32"][dtype] ?? dtype
            }
            let bytes = files.reduce(0.0) { sum, f in
                guard let path = f["path"] as? String, path.hasSuffix(".safetensors") else { return sum }
                return sum + (((f["lfs"] as? [String: Any])?["size"] as? NSNumber ?? f["size"] as? NSNumber)?.doubleValue ?? 0)
            }
            self.model.hfDetail = HFRepoDetail(
                id: id, architecture: cfg?["model_type"] as? String, contextTokens: ctx?.intValue,
                quants: bytes > 0 ? [HFQuant(tag: tag, sizeGB: bytes / 1_073_741_824.0)] : [],
                format: .mlx, vision: cfg?["vision_config"] != nil)
            self.model.hfError = bytes > 0 ? nil
                : (cfg == nil ? "Couldn't read \(id) — gated repos need `hf auth login`" : "No .safetensors weights in \(id)")
            self.pushUpdate()
        }
    }

    /// Architectures Splash's GGUF loader accepts (splash-1.1.0/install/gguf.py TEXT_MODEL_TYPES).
    static let splashArchitectures: Set<String> = ["qwen35", "qwen35moe"]

    /// Pull `hf.co/<repo>:<tag>` through Ollama's API with live progress, then load it.
    func pullToOllama(repo: String, tag: String) {
        let name = "hf.co/\(repo):\(tag)"
        pullTask?.cancel()
        model.hfJob = HFJob(target: .ollama, model: name, status: "Contacting Ollama…")
        pushUpdate()
        guard let url = URL(string: "\(ollamaBase)/api/pull") else { return }
        var req = URLRequest(url: url, timeoutInterval: 3600)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["model": name, "stream": true])
        pullTask = Task { [weak self] in
            func update(_ change: @escaping (inout HFJob) -> Void) {
                DispatchQueue.main.async {
                    guard let self, var job = self.model.hfJob, job.model == name else { return }
                    change(&job)
                    self.model.hfJob = job
                    self.pushUpdate()
                }
            }
            do {
                let (bytes, resp) = try await URLSession.shared.bytes(for: req)
                guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                    update { $0.failed = true; $0.status = "Ollama refused the pull (HTTP \((resp as? HTTPURLResponse)?.statusCode ?? 0))" }
                    return
                }
                var lastUpdate = Date.distantPast
                var weightsTotal = 0.0   // track the big weights layer; small template/params layers follow it
                for try await line in bytes.lines {
                    guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else { continue }
                    if let err = obj["error"] as? String {
                        update { $0.failed = true; $0.status = err }
                        return
                    }
                    let status = obj["status"] as? String ?? ""
                    let total = (obj["total"] as? NSNumber)?.doubleValue ?? 0
                    let done = (obj["completed"] as? NSNumber)?.doubleValue ?? 0
                    if total > 0, total < weightsTotal { continue }
                    weightsTotal = max(weightsTotal, total)
                    // throttle UI updates; always pass phase changes through
                    guard total == 0 || Date().timeIntervalSince(lastUpdate) > 0.25 else { continue }
                    lastUpdate = Date()
                    update { job in
                        if total > 0 {
                            job.fraction = done / total
                            job.status = String(format: "Downloading %.1f / %.1f GB", done / 1_073_741_824, total / 1_073_741_824)
                        } else {
                            job.fraction = status == "success" ? 1 : nil
                            job.status = status.prefix(1).uppercased() + status.dropFirst()
                        }
                    }
                }
                update { $0.done = true; $0.fraction = 1; $0.status = "Pulled — loading into RAM" }
                DispatchQueue.main.async { [weak self] in
                    self?.refreshOllama()
                    self?.loadModel(name)
                }
            } catch {
                let cancelled = (error as? URLError)?.code == .cancelled || error is CancellationError
                update { $0.failed = true; $0.status = cancelled ? "Cancelled" : error.localizedDescription }
            }
        }
    }

    // MARK: MLX from Hugging Face

    /// The venv that holds mlx-lm, mlx-vlm and huggingface_hub: the `mlxBinDir` setting, else the
    /// folder of a configured server, else a venv in a usual place.
    private var mlxBinDir: String {
        if let dir = UserDefaults.standard.string(forKey: "mlxBinDir"), !dir.isEmpty {
            return NSString(string: dir).expandingTildeInPath
        }
        if let dir = model.mlxServers.first(where: { $0.configured && !$0.binary.isEmpty })
            .map({ ($0.binary as NSString).deletingLastPathComponent }) { return dir }
        let found = AppPaths.firstExecutable(["~/.venv-mlx/bin/mlx_lm.server", "~/.venv/bin/mlx_lm.server",
                                             "~/mlx/bin/mlx_lm.server", "~/.local/pipx/venvs/mlx-lm/bin/mlx_lm.server"])
        return found.map { ($0 as NSString).deletingLastPathComponent } ?? NSString(string: "~/.venv-mlx/bin").expandingTildeInPath
    }
    /// Whether an MLX Python environment was found (the MLX section hides without one).
    var hasMlxEnv: Bool { FileManager.default.isExecutableFile(atPath: "\(mlxBinDir)/python") }
    static let panelMlxPort = 8081
    static let hfHubDir = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
    private var mlxDownload: Process?
    private var panelMlxProcess: Process?

    /// model_type values each server can load, read from the installed mlx_lm / mlx_vlm packages.
    lazy var mlxTypes: (lm: Set<String>, vlm: Set<String>) = {
        let fm = FileManager.default
        let lib = ((mlxBinDir as NSString).deletingLastPathComponent as NSString).appendingPathComponent("lib")
        guard let py = (try? fm.contentsOfDirectory(atPath: lib))?.first(where: { $0.hasPrefix("python") }) else { return ([], []) }
        let site = "\(lib)/\(py)/site-packages"
        let lm = (try? fm.contentsOfDirectory(atPath: "\(site)/mlx_lm/models")) ?? []
        let vlm = (try? fm.contentsOfDirectory(atPath: "\(site)/mlx_vlm/models")) ?? []
        return (Set(lm.filter { $0.hasSuffix(".py") }.map { String($0.dropLast(3)) }),
                Set(vlm.filter { !$0.contains(".") && !$0.hasPrefix("_") }))
    }()

    /// Which server runs a model type (vision models prefer mlx_vlm), or nil if neither knows it.
    func mlxServer(for modelType: String?, vision: Bool) -> String? {
        AppDelegate.mlxServer(types: mlxTypes, modelType: modelType, vision: vision)
    }

    static func mlxServer(types: (lm: Set<String>, vlm: Set<String>), modelType: String?, vision: Bool) -> String? {
        guard let t = modelType else { return "mlx_lm.server" }
        if vision, types.vlm.contains(t) { return "mlx_vlm.server" }
        if types.lm.contains(t) { return "mlx_lm.server" }
        if types.vlm.contains(t) { return "mlx_vlm.server" }
        return nil
    }

    /// Text models in the Hugging Face cache that look like MLX conversions the venv can serve.
    func loadMlxCached() {
        let hub = AppDelegate.hfHubDir
        let types = mlxTypes
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            let fm = FileManager.default
            var found: [MlxCachedModel] = []
            for dir in (try? fm.contentsOfDirectory(atPath: hub)) ?? [] where dir.hasPrefix("models--") {
                guard let sep = dir.dropFirst(8).range(of: "--") else { continue }
                let rest = dir.dropFirst(8)
                let repo = rest[..<sep.lowerBound] + "/" + rest[sep.upperBound...]
                if repo.range(of: "(?i)asr|tts|whisper|voice|audio|speech|parakeet|diariz", options: .regularExpression) != nil { continue }
                let snaps = "\(hub)/\(dir)/snapshots"
                let newest = ((try? fm.contentsOfDirectory(atPath: snaps)) ?? [])
                    .map { "\(snaps)/\($0)" }
                    .max { (try? fm.attributesOfItem(atPath: $0)[.modificationDate] as? Date ?? .distantPast) ?? .distantPast
                         < (try? fm.attributesOfItem(atPath: $1)[.modificationDate] as? Date ?? .distantPast) ?? .distantPast }
                guard let snap = newest,
                      let data = fm.contents(atPath: "\(snap)/config.json"),
                      let cfg = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
                let quantized = cfg["quantization"] != nil || cfg["quantization_config"] != nil
                guard quantized || repo.lowercased().contains("mlx") else { continue }
                let weights = ((try? fm.contentsOfDirectory(atPath: snap)) ?? []).filter { $0.hasSuffix(".safetensors") }
                guard !weights.isEmpty,
                      let server = AppDelegate.mlxServer(types: types, modelType: cfg["model_type"] as? String,
                                                         vision: cfg["vision_config"] != nil)
                else { continue }
                let bytes = weights.reduce(0.0) { sum, f in
                    let real = ("\(snap)/\(f)" as NSString).resolvingSymlinksInPath
                    return sum + ((try? fm.attributesOfItem(atPath: real)[.size] as? NSNumber)?.doubleValue ?? 0)
                }
                found.append(MlxCachedModel(repo: String(repo), path: snap, server: server, sizeGB: bytes / 1_073_741_824.0))
            }
            found.sort { $0.repo.lowercased() < $1.repo.lowercased() }
            DispatchQueue.main.async {
                self.model.mlxCached = found
                self.pushUpdate()
            }
        }
    }

    /// Runs huggingface_hub's snapshot_download in the MLX venv and prints byte progress as JSON lines.
    private static let hfDownloadScript = """
    import json, os, sys, time
    from tqdm import tqdm
    from huggingface_hub import snapshot_download

    class Report(tqdm):
        def __init__(self, *a, **k):
            k["disable"] = False
            k["file"] = open(os.devnull, "w")
            super().__init__(*a, **k)
        def refresh(self, *a, **k):
            if self.unit == "B" and time.monotonic() - getattr(self, "_t", 0.0) > 0.25:
                self._t = time.monotonic()
                print(json.dumps({"bar": self.desc, "done": self.n, "total": self.total or 0}), flush=True)
            return super().refresh(*a, **k)
        def update(self, n=1):
            r = super().update(n)
            self.refresh()
            return r

    print(json.dumps({"path": snapshot_download(sys.argv[1], tqdm_class=Report)}), flush=True)
    """

    /// Download `repo` into the Hugging Face cache (resumable), then serve it in the panel's MLX slot.
    func downloadToMlx(repo: String, sizeGB: Double, server: String) {
        cancelMlxDownload()
        model.hfJob = HFJob(target: .mlx, model: repo, status: "Starting download…")
        pushUpdate()
        runMlxDownload(repo: repo, sizeGB: sizeGB, server: server, xet: true)
    }

    private func runMlxDownload(repo: String, sizeGB: Double, server: String, xet: Bool) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "\(mlxBinDir)/python")
        p.arguments = ["-I", "-c", AppDelegate.hfDownloadScript, repo]
        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "HF_HUB_DISABLE_PROGRESS_BARS")
        if !xet { env["HF_HUB_DISABLE_XET"] = "1" }  // Xet sometimes dies with "Unable to parse string as hex hash"
        p.environment = env
        let out = Pipe(), err = Pipe()
        p.standardOutput = out
        p.standardError = err
        let expected = sizeGB * 1_073_741_824
        var buffer = Data(), errTail = Data(), path: String?, best = 0.0
        let lock = NSLock()
        out.fileHandleForReading.readabilityHandler = { [weak self] h in
            let chunk = h.availableData
            guard !chunk.isEmpty else { return }
            lock.lock(); buffer.append(chunk)
            var lines: [Data] = []
            while let nl = buffer.firstIndex(of: 0x0A) {
                lines.append(buffer[buffer.startIndex..<nl])
                buffer.removeSubrange(buffer.startIndex...nl)
            }
            lock.unlock()
            for line in lines {
                guard let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { continue }
                if let p = obj["path"] as? String { path = p; continue }
                let done = (obj["done"] as? NSNumber)?.doubleValue ?? 0
                // totals grow as files are discovered; measure against the repo size we already know
                let total = max((obj["total"] as? NSNumber)?.doubleValue ?? 0, expected)
                guard total > 0 else { continue }
                best = max(best, min(done / total, 0.99))
                let frac = best
                DispatchQueue.main.async {
                    guard let self, var job = self.model.hfJob, job.target == .mlx, job.model == repo, !job.failed else { return }
                    job.fraction = frac
                    job.status = String(format: "Downloading %.1f / %.1f GB", frac * total / 1_073_741_824, total / 1_073_741_824)
                    self.model.hfJob = job
                    self.pushUpdate()
                }
            }
        }
        err.fileHandleForReading.readabilityHandler = { h in
            let chunk = h.availableData
            lock.lock(); errTail.append(chunk); if errTail.count > 4096 { errTail = errTail.suffix(4096) }; lock.unlock()
        }
        p.terminationHandler = { [weak self] proc in
            out.fileHandleForReading.readabilityHandler = nil
            err.fileHandleForReading.readabilityHandler = nil
            let tail = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            if path == nil, let last = tail.split(separator: "\n").last,
               let obj = try? JSONSerialization.jsonObject(with: Data(last.utf8)) as? [String: Any] {
                path = obj["path"] as? String
            }
            lock.lock()
            let errText = String(data: errTail, encoding: .utf8) ?? ""
            lock.unlock()
            let status = proc.terminationStatus
            DispatchQueue.main.async {
                guard let self, let job = self.model.hfJob, job.target == .mlx, job.model == repo, !job.failed, !job.done
                else { return }  // cancelled or replaced
                self.mlxDownload = nil
                if status == 0, let path {
                    self.loadMlxCached()
                    self.serveInMlx(path: path, repo: repo, server: server)
                } else if xet, proc.terminationReason == .exit {
                    self.model.hfJob?.status = "Retrying without Xet…"
                    self.pushUpdate()
                    self.runMlxDownload(repo: repo, sizeGB: sizeGB, server: server, xet: false)
                } else {
                    let line = errText.split(separator: "\n").last { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
                    self.model.hfJob?.failed = true
                    self.model.hfJob?.status = line.map { String($0).trimmingCharacters(in: .whitespaces) } ?? "Download failed"
                    self.pushUpdate()
                }
            }
        }
        do {
            try p.run()
            mlxDownload = p
        } catch {
            model.hfJob?.failed = true
            model.hfJob?.status = "No MLX Python at \(mlxBinDir) — set mlxBinDir (see README)"
            pushUpdate()
        }
    }

    private func cancelMlxDownload() {
        mlxDownload?.terminate()
        mlxDownload = nil
    }

    /// Point the panel's MLX slot (:8081) at a model snapshot and start it; replaces what ran there.
    func serveInMlx(path: String, repo: String, server: String) {
        let port = AppDelegate.panelMlxPort
        notePanelIntent(.mlx, path)
        for s in model.mlxServers where s.running && s.port == port { notePanelIntent(.mlx, s.modelPath) }
        if model.hfJob?.model != repo || model.hfJob?.target != .mlx {
            model.hfJob = HFJob(target: .mlx, model: repo, status: "")
        }
        model.hfJob?.status = "Stopping the server on :\(port)…"
        model.hfJob?.fraction = nil
        pushUpdate()
        let pids = model.mlxServers.filter { $0.running && $0.port == port }.compactMap(\.pid)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            for pid in pids { kill(pid_t(pid), SIGTERM) }
            let deadline = Date().addingTimeInterval(10)
            while Date() < deadline, pids.contains(where: { kill(pid_t($0), 0) == 0 }) { usleep(200_000) }
            for pid in pids where kill(pid_t(pid), 0) == 0 { kill(pid_t(pid), SIGKILL) }
            DispatchQueue.main.async {
                guard let self, let job = self.model.hfJob, job.target == .mlx, job.model == repo else { return }
                let srv = self.writePanelMlxEntry(repo: repo, path: path, server: server)
                self.panelMlxProcess = self.launchMlx(srv)
                self.model.hfJob?.status = "Loading \(shortModelName(repo))…"
                self.model.hfJob?.started = Date()
                self.pushUpdate()
                self.watchPanelMlx(repo: repo)
            }
        }
    }

    /// Replace the panel's entry in mlx_servers.json, keeping hand-written entries untouched.
    private func writePanelMlxEntry(repo: String, path: String, server: String) -> MlxServerState {
        let port = AppDelegate.panelMlxPort
        let binary = "\(mlxBinDir)/\(server)"
        let args = ["--model", path, "--port", "\(port)", "--host", "127.0.0.1"]
        var arr = (try? Data(contentsOf: URL(fileURLWithPath: mlxConfigPath)))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [[String: Any]] } ?? []
        arr.removeAll { $0["panel"] as? Bool == true }
        arr.append(["name": shortModelName(repo), "port": port, "binary": binary, "args": args,
                    "keep_running": false, "panel": true])
        if let data = try? JSONSerialization.data(withJSONObject: arr, options: [.prettyPrinted, .withoutEscapingSlashes]) {
            try? data.write(to: URL(fileURLWithPath: mlxConfigPath))
        }
        loadMlxConfig()
        refreshMlx()
        return model.mlxServers.first { $0.panel }
            ?? MlxServerState(name: shortModelName(repo), port: port, binary: binary, args: args, panel: true)
    }

    /// Poll the slot until it answers /v1/models, or report the last log line if it exits.
    private func watchPanelMlx(repo: String) {
        let port = AppDelegate.panelMlxPort
        Task { [weak self] in
            while true {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                let alive = await MainActor.run { () -> Bool? in
                    guard let self, let job = self.model.hfJob, job.target == .mlx, job.model == repo,
                          !job.done, !job.failed else { return nil }
                    return self.panelMlxProcess?.isRunning ?? false
                }
                guard let alive else { return }
                if !alive {
                    let log = (try? String(contentsOfFile: AppDelegate.mlxLogPath(port: port), encoding: .utf8)) ?? ""
                    let last = log.split(separator: "\n").last { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
                    await MainActor.run {
                        self?.model.hfJob?.failed = true
                        self?.model.hfJob?.status = last.map(String.init) ?? "Server exited — see \(AppDelegate.mlxLogPath(port: port))"
                        self?.refreshMlx()
                    }
                    return
                }
                var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/models")!, timeoutInterval: 1)
                req.httpMethod = "GET"
                if let (_, resp) = try? await URLSession.shared.data(for: req), (resp as? HTTPURLResponse)?.statusCode == 200 {
                    await MainActor.run {
                        self?.model.hfJob?.done = true
                        self?.model.hfJob?.fraction = 1
                        self?.model.hfJob?.status = "Ready on :\(port)"
                        self?.refreshMlx()
                    }
                    return
                }
            }
        }
    }

    /// Cancel an Ollama pull, or stop a Splash model that is still starting; clears a finished job.
    func cancelHFJob() {
        guard let job = model.hfJob else { return }
        if !job.done, !job.failed {
            switch job.target {
            case .ollama:
                pullTask?.cancel()
            case .splash:
                if let slot = model.splashServers.first(where: { $0.label == AppDelegate.panelSplashLabel }) {
                    stopSplash(slot)
                }
            case .mlx:
                if mlxDownload != nil {
                    cancelMlxDownload()
                } else if let slot = model.mlxServers.first(where: { $0.panel && $0.running }) {
                    stopMlx(slot)
                } else {
                    panelMlxProcess?.terminate()
                }
            }
        }
        model.hfJob = nil
        pushUpdate()
    }

    static func formatCount(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0fK", Double(n) / 1_000) }
        return "\(n)"
    }

    static func formatTokens(_ n: Int) -> String {
        n >= 1024 && n % 1024 == 0 ? "\(n / 1024)K" : "\(n)"
    }

    static func formatDuration(_ seconds: Int) -> String {
        if seconds >= 3600 {
            return String(format: "%dh%02dm", seconds / 3600, (seconds % 3600) / 60)
        }
        if seconds >= 60 {
            return String(format: "%dm%02ds", seconds / 60, seconds % 60)
        }
        return "\(seconds)s"
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
}

// MARK: - Library: models on disk, favorites, history

enum Backend: String, Codable {
    case ollama, mlx, splash, hf

    var title: String {
        switch self {
        case .ollama: return "Ollama"
        case .mlx: return "MLX"
        case .splash: return "Splash"
        case .hf: return "HF cache"
        }
    }
    var badge: String {
        switch self {
        case .ollama: return "O"
        case .mlx: return "M"
        case .splash: return "S"
        case .hf: return "H"
        }
    }
}

/// One copy of a model in one backend.
struct LibraryInstance {
    let backend: Backend
    /// Ollama name · MLX snapshot path · Splash model id · "org/repo:TAG" for a GGUF in the HF cache
    let ref: String
    var repo: String? = nil
    var repoDir: String? = nil         // HF cache folder (MLX, HF, Splash target)
    let weightsBytes: Double
    /// Physical files this copy needs on disk (blob path → bytes). Shared files appear in several instances.
    let files: [String: Double]
    var server: String? = nil          // MLX: mlx_lm.server or mlx_vlm.server
    var assembly: String? = nil        // Splash: .resolved/<sha>
    var links: [String] = []           // Splash: install links pointing at the assembly
    var sourceRepos: [String] = []     // Splash: target + draft repos it pins
    var languageOnly = false           // Splash: installed without the vision tower
    var engine: String? = nil          // Splash: "Splish" when installed in a Splish checkout

    var id: String { "\(backend.rawValue)\(engine.map { "-" + $0.lowercased() } ?? "")|\(ref)" }
    /// Where this copy lives, as the user knows it ("Splish" rather than "Splash" for a checkout).
    var place: String { engine ?? backend.title }
}

/// A model as the user thinks of it: the same weights, wherever they are installed.
struct LibraryModel {
    let key: String
    var name: String
    var repo: String?
    var instances: [LibraryInstance]
    var sizeBytes: Double { instances.map(\.weightsBytes).max() ?? 0 }
}

/// One model whose weights sit on disk twice: once in Ollama's blobs, once in the HF cache.
struct DuplicateGroup {
    let key: String                    // library model key
    let name: String
    let bytes: Double                  // space a single copy would give back
    let ollamaNames: [String]          // Ollama models using the Ollama copy
    let hfUsers: [Backend]             // what uses the HF-cache copy (.splash, .mlx, or .hf = nothing else)

    /// The one-line fix, in plain words.
    var advice: String {
        if hfUsers.allSatisfy({ $0 == .hf }) { return "HF-cache copy isn't used by anything — delete it to free the space" }
        let users = hfUsers.filter { $0 != .hf }.map(\.title).joined(separator: " and ")
        return "Both copies in use: Ollama, and \(users) through the HF cache"
    }
}

struct LibrarySnapshot {
    var models: [LibraryModel]
    var diskBytes: Double              // every physical file counted once
    var duplicates: [DuplicateGroup]
    var freeBytes: Double
    var scannedAt: Date

    var duplicateBytes: Double { duplicates.reduce(0) { $0 + $1.bytes } }
    var instances: [LibraryInstance] { models.flatMap(\.instances) }
}

enum LibraryScanner {
    static var ollamaModelsDir: String {
        if let dir = ProcessInfo.processInfo.environment["OLLAMA_MODELS"], !dir.isEmpty { return dir }
        return NSString(string: "~/.ollama/models").expandingTildeInPath
    }

    /// Repos in the HF cache that belong to other tools (speech, image models).
    static let nonLLM = "(?i)asr|tts|whisper|voice|audio|speech|parakeet|diariz|flux|qwen-image|chatterbox|timesfm|s1-mini|fish-audio"

    struct HubFile {
        let name: String       // path inside the snapshot
        let link: String       // snapshot symlink
        let blob: String       // resolved blob
        let bytes: Double
    }

    static func fileSize(_ path: String) -> Double {
        (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.doubleValue ?? 0
    }

    static func repoID(fromDir dir: String) -> String? {
        let name = (dir as NSString).lastPathComponent
        guard name.hasPrefix("models--") else { return nil }
        let rest = name.dropFirst(8)
        guard let sep = rest.range(of: "--") else { return nil }
        return rest[..<sep.lowerBound] + "/" + rest[sep.upperBound...]
    }

    static func repoDir(_ repo: String) -> String {
        "\(AppDelegate.hfHubDir)/models--\(repo.replacingOccurrences(of: "/", with: "--"))"
    }

    /// Every snapshot symlink in a cached repo, with the blob it points at.
    static func hubFiles(repoDir: String) -> [HubFile] {
        let fm = FileManager.default
        let snaps = "\(repoDir)/snapshots"
        var out: [HubFile] = []
        for snap in (try? fm.contentsOfDirectory(atPath: snaps)) ?? [] {
            let root = "\(snaps)/\(snap)"
            guard let walker = fm.enumerator(atPath: root) else { continue }
            while let rel = walker.nextObject() as? String {
                let link = "\(root)/\(rel)"
                guard (try? fm.attributesOfItem(atPath: link))?[.type] as? FileAttributeType == .typeSymbolicLink
                else { continue }
                let blob = (link as NSString).resolvingSymlinksInPath
                guard blob != link, fm.fileExists(atPath: blob) else { continue }
                out.append(HubFile(name: rel, link: link, blob: blob, bytes: fileSize(blob)))
            }
        }
        return out
    }

    static func newestSnapshot(repoDir: String) -> String? {
        let fm = FileManager.default
        let snaps = "\(repoDir)/snapshots"
        func modified(_ p: String) -> Date { (try? fm.attributesOfItem(atPath: p)[.modificationDate] as? Date) ?? .distantPast }
        return ((try? fm.contentsOfDirectory(atPath: snaps)) ?? []).map { "\(snaps)/\($0)" }.max { modified($0) < modified($1) }
    }

    /// "UD-Q4_K_M" from "…/Qwen3.8-27B-UD-Q4_K_M-00001-of-00002.gguf"; nil for projectors, imatrix, MTP heads.
    static func ggufQuantTag(_ path: String) -> String? {
        let lower = path.lowercased()
        guard lower.hasSuffix(".gguf") else { return nil }
        let file = (lower as NSString).lastPathComponent
        if lower.contains("mmproj") || lower.contains("imatrix") || lower.hasPrefix("mtp/") || file.hasPrefix("mtp-") { return nil }
        var stem = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        stem = stem.replacingOccurrences(of: "-[0-9]{5}-of-[0-9]{5}$", with: "", options: .regularExpression)
        return stem.range(of: "(?i)(UD-)?([A-Z]*Q[0-9]+(_[A-Z0-9]+)*|BF16|F16|F32|[A-Z]*FP[0-9]+(_[A-Z0-9]+)*)$",
                          options: .regularExpression).map { String(stem[$0]) } ?? stem
    }

    /// Splash install links: models/<org>/<repo:VARIANT> and .selections/<hash>.
    static func splashLinks() -> [String] {
        let fm = FileManager.default
        var links: [String] = []
        for dir in AppDelegate.splashModelRoots {
        for org in (try? fm.contentsOfDirectory(atPath: dir)) ?? [] where !org.hasPrefix(".") || org == ".selections" {
            let orgDir = "\(dir)/\(org)"
            for name in (try? fm.contentsOfDirectory(atPath: orgDir)) ?? [] where !name.hasPrefix(".") {
                let link = "\(orgDir)/\(name)"
                if (try? fm.attributesOfItem(atPath: link))?[.type] as? FileAttributeType == .typeSymbolicLink {
                    links.append(link)
                }
            }
        }
        }
        return links
    }

    /// Bytes of a file not shared with an APFS clone (a clone-merged twin costs nothing extra).
    static func privateBytes(_ path: String) -> Double {
        cloneInfo(path)?.privateBytes ?? fileSize(path)
    }

    /// APFS private size and clone family of a file (clones of one another share the clone ID).
    static func cloneInfo(_ path: String) -> (privateBytes: Double, cloneID: UInt64)? {
        var attrs = attrlist()
        attrs.bitmapcount = u_short(ATTR_BIT_MAP_COUNT)
        attrs.forkattr = 0x0000_0108   // ATTR_CMNEXT_PRIVATESIZE | ATTR_CMNEXT_CLONEID, with FSOPT_ATTR_CMN_EXTENDED
        var buf = [UInt8](repeating: 0, count: 32)
        guard path.withCString({ getattrlist($0, &attrs, &buf, buf.count, 0x20) }) == 0 else { return nil }
        return buf.withUnsafeBytes {
            (Double($0.loadUnaligned(fromByteOffset: 4, as: Int64.self)), $0.loadUnaligned(fromByteOffset: 12, as: UInt64.self))
        }
    }

    /// Splash's prepared copies of model weights (one folder per tensor group), keyed by the
    /// assembly folder they were prepared from. Not removed by Splash's own garbage collection.
    static let splashWeightCache = NSString(string: "~/Library/Caches/Splash/weights").expandingTildeInPath

    static func splashPreparedWeights() -> [String: [String: Double]] {
        let fm = FileManager.default
        var byAssembly: [String: [String: Double]] = [:]
        for entry in (try? fm.contentsOfDirectory(atPath: splashWeightCache)) ?? [] {
            let dir = "\(splashWeightCache)/\(entry)"
            guard let text = try? String(contentsOfFile: "\(dir)/source", encoding: .utf8),
                  let line = text.split(separator: "\n").first(where: { $0.hasPrefix("source ") }),
                  let range = line.range(of: "/.resolved/") else { continue }
            let rest = line[range.upperBound...]
            let assembly = String(line.dropFirst(7)[..<range.lowerBound]) + "/.resolved/"
                + String(rest.prefix { $0 != "/" })
            byAssembly[assembly, default: [:]]["\(dir)/weights"] = fileSize("\(dir)/weights")
        }
        return byAssembly
    }

    static func readJSON(_ path: String) -> [String: Any]? {
        FileManager.default.contents(atPath: path).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
    }

    static func scan(serverFor: (String?, Bool) -> String?) -> LibrarySnapshot {
        let fm = FileManager.default
        let hub = AppDelegate.hfHubDir
        var models: [String: LibraryModel] = [:]
        var order: [String] = []
        func add(_ key: String, name: String, repo: String?, _ inst: LibraryInstance) {
            if models[key] == nil {
                models[key] = LibraryModel(key: key, name: name, repo: repo, instances: [])
                order.append(key)
            }
            models[key]!.instances.append(inst)
        }

        // HF cache: every repo's files, and LFS blobs by sha256 (Ollama names its blobs the same way)
        var repoFiles: [(repo: String, dir: String, files: [HubFile])] = []
        var hfBlobBySha: [String: String] = [:]
        for name in ((try? fm.contentsOfDirectory(atPath: hub)) ?? []).sorted() where name.hasPrefix("models--") {
            let dir = "\(hub)/\(name)"
            guard let repo = repoID(fromDir: dir) else { continue }
            let files = hubFiles(repoDir: dir)
            repoFiles.append((repo, dir, files))
            for f in files {
                let sha = (f.blob as NSString).lastPathComponent
                if sha.count == 64 { hfBlobBySha[sha] = f.blob }
            }
        }

        // 1. Splash installs (they link into the HF cache; no weights of their own)
        var linksByAssembly: [String: [String]] = [:]
        for link in splashLinks() {
            linksByAssembly[(link as NSString).resolvingSymlinksInPath, default: []].append(link)
        }
        var splashBlobs = Set<String>()
        let prepared = splashPreparedWeights()
        for (assembly, links) in linksByAssembly.sorted(by: { $0.key < $1.key }) {
            guard let record = readJSON("\(assembly)/model.json"), let id = record["model"] as? String else { continue }
            var files: [String: Double] = [:]
            var weights = 0.0
            for (name, entry) in record["files"] as? [String: Any] ?? [:] {
                guard let path = (entry as? [String: Any])?["path"] as? String, path.hasPrefix(hub + "/") else { continue }
                let blob = (path as NSString).resolvingSymlinksInPath
                let bytes = fileSize(blob)
                files[blob] = bytes
                if name.hasPrefix("target/") { weights += bytes }
            }
            splashBlobs.formUnion(files.keys)
            files.merge(prepared[assembly] ?? [:]) { a, _ in a }
            let sources = record["sources"] as? [String: Any] ?? [:]
            let repos = ["target", "draft"].compactMap { (sources[$0] as? [String: Any])?["repo"] as? String }
            let target = repos.first
            add(id.lowercased(), name: shortModelName(id), repo: target, LibraryInstance(
                backend: .splash, ref: id, repo: target, repoDir: target.map(repoDir), weightsBytes: weights,
                files: files, assembly: assembly, links: links.sorted(), sourceRepos: repos,
                languageOnly: record["vision_format"] as? String == "none",
                engine: AppDelegate.splashEngine(for: assembly) == "Splish" ? "Splish" : nil))
        }

        // 2. HF cache: GGUF quants not installed in Splash, and MLX conversions
        for (repo, dir, files) in repoFiles {
            if repo.range(of: nonLLM, options: .regularExpression) != nil { continue }
            if files.contains(where: { splashBlobs.contains($0.blob) && !$0.name.lowercased().hasSuffix(".gguf") }) {
                continue  // a Splash draft model (DFlash), shown as part of its install
            }
            let ggufs = files.filter { $0.name.lowercased().hasSuffix(".gguf") }
            if !ggufs.isEmpty {
                var groups: [String: [HubFile]] = [:]
                for f in ggufs { if let tag = ggufQuantTag(f.name) { groups[tag, default: []].append(f) } }
                for (tag, group) in groups.sorted(by: { $0.key < $1.key }) {
                    if group.contains(where: { splashBlobs.contains($0.blob) }) { continue }
                    var gf: [String: Double] = [:]
                    for f in group { gf[f.blob] = f.bytes }
                    let ref = "\(repo):\(tag)"
                    add(ref.lowercased(), name: "\(shortModelName(repo)):\(tag)", repo: repo, LibraryInstance(
                        backend: .hf, ref: ref, repo: repo, repoDir: dir,
                        weightsBytes: gf.values.reduce(0, +), files: gf))
                }
                continue
            }
            guard let snap = newestSnapshot(repoDir: dir), let cfg = readJSON("\(snap)/config.json") else { continue }
            let quantized = cfg["quantization"] != nil || cfg["quantization_config"] != nil
            guard quantized || repo.lowercased().contains("mlx") else { continue }
            let weights = files.filter { $0.name.hasSuffix(".safetensors") && $0.link.hasPrefix(snap + "/") }
            guard !weights.isEmpty,
                  let server = serverFor(cfg["model_type"] as? String, cfg["vision_config"] != nil) else { continue }
            var all: [String: Double] = [:]
            let blobsDir = "\(dir)/blobs"
            for b in (try? fm.contentsOfDirectory(atPath: blobsDir)) ?? [] {
                all["\(blobsDir)/\(b)"] = fileSize("\(blobsDir)/\(b)")
            }
            add(repo.lowercased(), name: shortModelName(repo), repo: repo, LibraryInstance(
                backend: .mlx, ref: snap, repo: repo, repoDir: dir,
                weightsBytes: weights.reduce(0) { $0 + $1.bytes }, files: all, server: server))
        }

        // 3. Ollama manifests: hf.co pulls and aliases merge with the HF entry they came from
        let ollama = ollamaModelsDir
        let manifests = "\(ollama)/manifests"
        var twins: [String: (ollamaBlob: String, hfBlob: String)] = [:]   // sha → the two physical copies
        if let walker = fm.enumerator(atPath: manifests) {
            while let rel = walker.nextObject() as? String {
                let parts = rel.split(separator: "/").map(String.init)
                guard parts.count == 4, let manifest = readJSON("\(manifests)/\(rel)"),
                      let layers = manifest["layers"] as? [[String: Any]] else { continue }
                let name = parts[0] == "registry.ollama.ai"
                    ? (parts[1] == "library" ? "\(parts[2]):\(parts[3])" : "\(parts[1])/\(parts[2]):\(parts[3])")
                    : "\(parts[0])/\(parts[1])/\(parts[2]):\(parts[3])"
                var files: [String: Double] = [:]
                var modelLayer: [String: Any]?
                for layer in layers + [manifest["config"] as? [String: Any] ?? [:]] {
                    guard let digest = layer["digest"] as? String else { continue }
                    let blob = "\(ollama)/blobs/\(digest.replacingOccurrences(of: ":", with: "-"))"
                    files[blob] = (layer["size"] as? NSNumber)?.doubleValue ?? fileSize(blob)
                    if (layer["mediaType"] as? String)?.hasSuffix("image.model") == true, modelLayer == nil { modelLayer = layer }
                }
                // GGUF models carry one image.model layer; MLX-format models carry many image.tensor layers
                let tensors = layers.filter { ($0["mediaType"] as? String)?.hasSuffix("image.tensor") == true }
                guard modelLayer != nil || !tensors.isEmpty else { continue }
                let sha = (modelLayer?["digest"] as? String).map { String($0.dropFirst("sha256:".count)) } ?? ""
                let weights = modelLayer.map { ($0["size"] as? NSNumber)?.doubleValue ?? 0 }
                    ?? tensors.reduce(0) { $0 + (($1["size"] as? NSNumber)?.doubleValue ?? 0) }
                var key = "ollama:\(name.lowercased())"
                var display = shortModelName(name)
                var repo: String?
                if name.hasPrefix("hf.co/") {
                    key = String(name.dropFirst(6)).lowercased()
                    repo = String(name.dropFirst(6).split(separator: ":").first ?? "")
                } else if let from = modelLayer?["from"] as? String, from.hasPrefix("hf.co/") {
                    key = String(from.dropFirst(6)).lowercased()
                    repo = String(from.dropFirst(6).split(separator: ":").first ?? "")
                } else if let blob = hfBlobBySha[sha],
                          let owner = order.first(where: { k in models[k]!.instances.contains { $0.files[blob] != nil } }) {
                    key = owner
                }
                if let existing = models[key] { display = existing.name; repo = repo ?? existing.repo }
                else if let repo, let tag = key.split(separator: ":").last { display = "\(shortModelName(repo)):\(tag.uppercased())" }
                if let hfBlob = hfBlobBySha[sha] {
                    twins[sha] = ("\(ollama)/blobs/sha256-\(sha)", hfBlob)
                }
                add(key, name: display, repo: repo, LibraryInstance(
                    backend: .ollama, ref: name, repo: repo, weightsBytes: weights, files: files))
            }
        }

        var unique: [String: Double] = [:]
        for m in models.values { for i in m.instances { for (p, b) in i.files { unique[p] = b } } }
        let free = (try? URL(fileURLWithPath: NSHomeDirectory())
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage.map(Double.init) ?? 0

        // Twins grouped per model; APFS clones share their blocks, so only private bytes count
        var duplicates: [String: DuplicateGroup] = [:]
        var cloneShared = 0.0
        for (_, t) in twins.sorted(by: { $0.key < $1.key }) {
            let bytes = min(privateBytes(t.ollamaBlob), privateBytes(t.hfBlob))
            cloneShared += fileSize(t.hfBlob) - bytes   // blocks both paths share are on disk once
            guard bytes > 64 * 1_048_576,
                  let m = models.values.first(where: { $0.instances.contains { $0.files[t.ollamaBlob] != nil } })
            else { continue }
            let ollamaNames = m.instances.filter { $0.backend == .ollama && $0.files[t.ollamaBlob] != nil }.map(\.ref)
            let hfUsers = m.instances.filter { $0.backend != .ollama && $0.files[t.hfBlob] != nil }.map(\.backend)
            let prior = duplicates[m.key]
            duplicates[m.key] = DuplicateGroup(
                key: m.key, name: m.name, bytes: (prior?.bytes ?? 0) + bytes,
                ollamaNames: Array(Set((prior?.ollamaNames ?? []) + ollamaNames)).sorted(),
                hfUsers: Array(Set((prior?.hfUsers ?? []) + hfUsers)).sorted { $0.rawValue < $1.rawValue })
        }
        return LibrarySnapshot(models: order.compactMap { models[$0] },
                               diskBytes: unique.values.reduce(0, +) - cloneShared,
                               duplicates: duplicates.values.sorted { $0.bytes > $1.bytes },
                               freeBytes: free, scannedAt: Date())
    }

    // MARK: delete

    enum DeleteError: LocalizedError {
        case failed(String)
        var errorDescription: String? { if case let .failed(s) = self { return s }; return nil }
    }

    /// Bytes that actually leave the disk when `targets` go: files no other copy still needs.
    static func bytesFreed(_ targets: [LibraryInstance], in snap: LibrarySnapshot) -> Double {
        let ids = Set(targets.map(\.id))
        var kept = Set<String>()
        for i in snap.instances where !ids.contains(i.id) { kept.formUnion(i.files.keys) }
        var gone: [String: Double] = [:]
        for t in targets { for (p, b) in t.files where !kept.contains(p) { gone[p] = b } }
        // APFS clones share blocks: those come back only when every clone of them goes
        var total = 0.0
        var families: [UInt64: (shared: Double, members: Int)] = [:]
        for (p, b) in gone {
            guard let c = cloneInfo(p), c.privateBytes < b - 1_048_576 else { total += b; continue }
            total += c.privateBytes
            let f = families[c.cloneID]
            families[c.cloneID] = (max(f?.shared ?? 0, b - c.privateBytes), (f?.members ?? 0) + 1)
        }
        for f in families.values where f.members > 1 { total += f.shared }
        return total
    }

    /// Remove blobs from the HF cache with their snapshot links; a repo with no weights
    /// and no Splash pins left is removed entirely.
    static func removeHubBlobs(_ blobs: Set<String>) throws {
        let fm = FileManager.default
        let hub = AppDelegate.hfHubDir
        let safe = blobs.filter { $0.hasPrefix(hub + "/models--") && $0.contains("/blobs/") }
        let byRepo = Dictionary(grouping: safe) { (($0 as NSString).deletingLastPathComponent as NSString).deletingLastPathComponent }
        for (dir, list) in byRepo {
            let set = Set(list)
            for f in hubFiles(repoDir: dir) where set.contains(f.blob) { try? fm.removeItem(atPath: f.link) }
            for blob in list where fm.fileExists(atPath: blob) {
                do { try fm.removeItem(atPath: blob) } catch { throw DeleteError.failed("Couldn't remove \(blob): \(error.localizedDescription)") }
            }
            // projectors (mmproj) alone don't keep a repo alive
            let weightsLeft = hubFiles(repoDir: dir).contains { ggufQuantTag($0.name) != nil || $0.name.hasSuffix(".safetensors") }
            let pins = (try? fm.contentsOfDirectory(atPath: "\(dir)/refs/splash")) ?? []
            if !weightsLeft, pins.isEmpty { try? fm.removeItem(atPath: dir) }
        }
    }

    /// Splash's own pin owner: sha256 of the resolved install folder + link name (install/hub.py pin_owner).
    static func splashPinOwner(_ link: String) -> String {
        let parent = ((link as NSString).deletingLastPathComponent as NSString).resolvingSymlinksInPath
        let owner = "\(parent)/\((link as NSString).lastPathComponent)"
        return SHA256.hash(data: Data(owner.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// Unlink a Splash install, drop its pins and its assembly. Runs under Splash's install lock.
    static func removeSplashInstall(_ inst: LibraryInstance) throws {
        let fm = FileManager.default
        let place = inst.assembly ?? inst.links.first ?? ""
        let dir = AppDelegate.splashModelRoots.first { place.hasPrefix($0 + "/") } ?? AppDelegate.splashModelsDir
        let lockPath = "\(dir)/.install.lock"
        let fd = open(lockPath, O_RDWR | O_CREAT, 0o644)
        guard fd >= 0 else { throw DeleteError.failed("Couldn't open Splash's install lock") }
        defer { close(fd) }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            throw DeleteError.failed("Splash is installing a model — try again when it's done")
        }
        defer { flock(fd, LOCK_UN) }

        for link in inst.links where link.hasPrefix(dir + "/") {
            let owner = splashPinOwner(link)
            for repo in inst.sourceRepos {
                let pins = "\(repoDir(repo))/refs/splash"
                try? fm.removeItem(atPath: "\(pins)/\(owner)")
                if ((try? fm.contentsOfDirectory(atPath: pins)) ?? []).isEmpty { try? fm.removeItem(atPath: pins) }
            }
            try? fm.removeItem(atPath: link)
        }
        guard let assembly = inst.assembly, assembly.hasPrefix(dir + "/.resolved/") else { return }
        let metadata = readJSON("\(assembly)/model.json")?["metadata"] as? String
        try? fm.removeItem(atPath: assembly)
        // Splash's prepared copy of this install's weights (read-only files; the folder is ours to remove)
        for weights in (splashPreparedWeights()[assembly] ?? [:]).keys {
            let entry = (weights as NSString).deletingLastPathComponent
            guard entry.hasPrefix(splashWeightCache + "/") else { continue }
            for f in (try? fm.contentsOfDirectory(atPath: entry)) ?? [] {
                chmod("\(entry)/\(f)", 0o600)
            }
            try? fm.removeItem(atPath: entry)
        }
        if let metadata {
            let others = ((try? fm.contentsOfDirectory(atPath: "\(dir)/.resolved")) ?? [])
                .compactMap { readJSON("\(dir)/.resolved/\($0)/model.json")?["metadata"] as? String }
            if !others.contains(metadata) { try? fm.removeItem(atPath: "\(dir)/.metadata/\(metadata)") }
        }
    }
}

/// Favorites and usage totals (library.json) plus an append-only event log (history.jsonl).
final class LibraryStore {
    struct Stats: Codable {
        var name: String?
        var favorite = false
        var favoritedAt: Date?
        var firstSeen: Date?
        var lastUsed: Date?
        var lastBackend: Backend?
        var lastRef: String?
        var loadCount = 0
        var secondsLoaded = 0.0
        var peakGB = 0.0
        var deletedAt: Date?
    }
    private struct File: Codable {
        var version = 1
        var models: [String: Stats] = [:]
    }

    static let dir = AppPaths.stateDir
    private let libraryPath = "\(LibraryStore.dir)/library.json"
    private let historyPath = "\(LibraryStore.dir)/history.jsonl"
    private(set) var models: [String: Stats] = [:]

    init() {
        try? FileManager.default.createDirectory(atPath: LibraryStore.dir, withIntermediateDirectories: true)
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .iso8601
        if let data = FileManager.default.contents(atPath: libraryPath), let file = try? dec.decode(File.self, from: data) {
            models = file.models
        }
        trimHistory()
    }

    func stats(_ key: String) -> Stats? { models[key] }
    func isFavorite(_ key: String) -> Bool { models[key]?.favorite ?? false }
    var favorites: Set<String> { Set(models.filter { $0.value.favorite }.keys) }

    func update(_ key: String, name: String?, _ change: (inout Stats) -> Void) {
        var s = models[key] ?? Stats(firstSeen: Date())
        if let name { s.name = name }
        change(&s)
        models[key] = s
        save()
    }

    /// New models get a first-seen date so "unused for 30 days" means something.
    func noteSeen(_ keys: [(String, String)]) {
        var changed = false
        for (key, name) in keys where models[key] == nil {
            models[key] = Stats(name: name, firstSeen: Date())
            changed = true
        }
        if changed { save() }
    }

    /// Debug windows (`--window`) look but don't touch: no favorites, stats or history are written.
    static let readOnly = CommandLine.arguments.contains("--window")

    func save() {
        guard !LibraryStore.readOnly else { return }
        let enc = JSONEncoder()
        enc.dateEncodingStrategy = .iso8601
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? enc.encode(File(models: models)) {
            try? data.write(to: URL(fileURLWithPath: libraryPath), options: .atomic)
        }
    }

    func log(_ event: String, _ fields: [String: Any]) {
        guard !LibraryStore.readOnly else { return }
        var obj = fields
        obj["ev"] = event
        obj["t"] = ISO8601DateFormatter().string(from: Date())
        guard var line = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes])
        else { return }
        line.append(0x0A)
        if let h = FileHandle(forWritingAtPath: historyPath) {
            h.seekToEndOfFile()
            h.write(line)
            try? h.close()
        } else {
            FileManager.default.createFile(atPath: historyPath, contents: line)
        }
    }

    /// Keep 180 days of events, at most ~5 MB.
    private func trimHistory() {
        guard let text = try? String(contentsOfFile: historyPath, encoding: .utf8) else { return }
        let cutoff = ISO8601DateFormatter().string(from: Date().addingTimeInterval(-180 * 86_400))
        var lines = text.split(separator: "\n").filter { line in
            guard let r = line.range(of: "\"t\":\"") else { return true }
            return line[r.upperBound...].prefix(20) >= cutoff.prefix(20)
        }
        var bytes = lines.reduce(0) { $0 + $1.utf8.count + 1 }
        while bytes > 5_000_000, !lines.isEmpty { bytes -= lines.removeFirst().utf8.count + 1 }
        let out = lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")
        if out.utf8.count != text.utf8.count { try? out.write(toFile: historyPath, atomically: true, encoding: .utf8) }
    }
}


// MARK: - Library actions, sessions, history

/// A model seen in memory, from the first poll that shows it until it has been gone for 5 s.
struct LiveSession {
    let key: String
    let name: String
    let backend: Backend
    let ref: String
    let start: Date
    var peakGB: Double
    var expiresAt: Date?
    var missingSince: Date?
}

/// JSON numbers like 0.3 instead of 0.29999999999999999.
func oneDecimal(_ x: Double) -> NSDecimalNumber { NSDecimalNumber(string: String(format: "%.1f", x)) }

struct RecentItem {
    let key: String
    let name: String
    let instance: LibraryInstance
    let lastUsed: Date?
    let favorite: Bool
}

extension AppDelegate {
    // MARK: scan

    func rescanLibrary() {
        guard !libraryScanInFlight else { libraryRescanQueued = true; return }
        libraryScanInFlight = true
        let types = mlxTypes
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let snap = LibraryScanner.scan { modelType, vision in
                AppDelegate.mlxServer(types: types, modelType: modelType, vision: vision)
            }
            DispatchQueue.main.async {
                guard let self else { return }
                self.model.library = snap
                self.libraryScanInFlight = false
                self.store.noteSeen(snap.models.map { ($0.key, $0.name) })
                self.pushUpdate()
                if self.libraryRescanQueued {
                    self.libraryRescanQueued = false
                    self.rescanLibrary()
                }
            }
        }
    }

    /// The library entry a running model belongs to (falls back to the same naming rules before a scan).
    func libraryKey(_ backend: Backend, _ ref: String) -> (key: String, name: String) {
        let lower = ref.lowercased()
        for m in model.library?.models ?? [] {
            for i in m.instances where i.backend == backend {
                let match = backend == .mlx
                    ? (i.repoDir.map { ref.hasPrefix($0 + "/") || ref == $0 } ?? false)
                    : i.ref.lowercased() == lower
                if match { return (m.key, m.name) }
            }
        }
        switch backend {
        case .ollama:
            return ref.hasPrefix("hf.co/") ? (String(lower.dropFirst(6)), shortModelName(ref)) : ("ollama:\(lower)", shortModelName(ref))
        case .mlx:
            let dir = ref.components(separatedBy: "/snapshots/").first ?? ref
            let repo = LibraryScanner.repoID(fromDir: dir) ?? ref
            return (repo.lowercased(), shortModelName(repo))
        case .splash, .hf:
            return (lower, shortModelName(ref))
        }
    }

    func isFavorite(_ backend: Backend, _ ref: String) -> Bool {
        store.isFavorite(libraryKey(backend, ref).key)
    }

    func toggleFavorite(key: String, name: String) {
        store.update(key, name: name) { s in
            s.favorite.toggle()
            s.favoritedAt = s.favorite ? Date() : nil
        }
        store.log(store.isFavorite(key) ? "favorite" : "unfavorite", ["key": key])
        pushUpdate()
    }

    func notePanelIntent(_ backend: Backend, _ ref: String?) {
        guard let ref else { return }
        panelIntents["\(backend.rawValue)|\(ref)"] = Date()
    }

    // MARK: sessions → history

    /// Diff what is in memory now against the last poll and record loads and unloads.
    func trackSessions() {
        let now = Date()
        var live: [String: (backend: Backend, ref: String, gb: Double, expires: Date?)] = [:]
        if model.ollamaUp {
            for m in model.loaded { live["ollama|\(m.name)"] = (.ollama, m.name, m.sizeGB, m.expiresAt) }
        }
        for s in model.mlxServers where s.running {
            if let p = s.modelPath { live["mlx|\(p)"] = (.mlx, p, s.weightsGB > 0 ? s.weightsGB : s.rssGB, nil) }
        }
        for s in model.splashServers where s.running && s.ready {
            if let m = s.model { live["splash|\(m)"] = (.splash, m, s.memGB, nil) }
        }

        for (id, v) in live {
            if var s = sessions[id] {
                s.peakGB = max(s.peakGB, v.gb)
                s.expiresAt = v.expires
                s.missingSince = nil
                sessions[id] = s
                continue
            }
            let (key, name) = libraryKey(v.backend, v.ref)
            let observed = now.timeIntervalSince(launchedAt) < 15
            let byPanel = panelIntents[id].map { now.timeIntervalSince($0) < 900 } ?? false
            sessions[id] = LiveSession(key: key, name: name, backend: v.backend, ref: v.ref, start: now,
                                       peakGB: v.gb, expiresAt: v.expires)
            store.update(key, name: name) { st in
                st.lastUsed = now
                st.lastBackend = v.backend
                st.lastRef = v.ref
                st.deletedAt = nil
                if !observed { st.loadCount += 1 }
            }
            if !observed {
                store.log("load", ["key": key, "backend": v.backend.rawValue, "ref": v.ref,
                                   "by": byPanel ? "panel" : "external", "gb": oneDecimal(v.gb)])
            }
            panelIntents[id] = nil
        }

        for (id, var s) in Array(sessions) where live[id] == nil {
            guard let since = s.missingSince else {
                s.missingSince = now
                sessions[id] = s
                continue
            }
            guard now.timeIntervalSince(since) > 5 else { continue }
            let duration = since.timeIntervalSince(s.start)
            let byPanel = panelIntents[id].map { now.timeIntervalSince($0) < 120 } ?? false
            let idle = s.expiresAt.map { $0.timeIntervalSince(since) < 10 } ?? false
            store.update(s.key, name: s.name) { st in
                st.lastUsed = since
                st.secondsLoaded += duration
                st.peakGB = max(st.peakGB, s.peakGB)
            }
            store.log("unload", ["key": s.key, "backend": s.backend.rawValue, "ref": s.ref,
                                 "by": byPanel ? "panel" : (idle ? "idle" : "external"),
                                 "durationS": Int(duration), "peakGB": oneDecimal(s.peakGB)])
            sessions[id] = nil
            panelIntents[id] = nil
        }
    }

    /// Downloads and model starts started from the panel, logged once per state change.
    func logJobTransitions() {
        guard let job = model.hfJob else { return }
        let state = "\(job.target.rawValue)|\(job.model)|\(job.done)|\(job.failed)|\(job.started != nil)"
        guard state != lastJobLogged else { return }
        let firstSight = !lastJobLogged.hasPrefix("\(job.target.rawValue)|\(job.model)|")
        lastJobLogged = state
        let fields: [String: Any] = ["backend": job.target.rawValue.lowercased(), "ref": job.model]
        if job.failed {
            var f = fields
            f["error"] = job.status
            store.log(job.started != nil || job.target == .splash ? "load_fail" : "download_fail", f)
        } else if job.done {
            store.log(job.target == .ollama ? "download_done" : "serve_ready", fields)
            rescanLibrary()
        } else if firstSight {
            store.log(job.target == .splash ? "serve_start" : "download_start", fields)
        } else if job.target == .mlx, job.started != nil {
            store.log("download_done", fields)
        }
    }

    // MARK: recent

    func liveKeys() -> Set<String> { Set(sessions.values.map(\.key)) }

    /// The copy to load by default: the one used last, else Ollama → MLX → Splash → HF cache.
    func preferredInstance(_ m: LibraryModel) -> LibraryInstance? {
        let st = store.stats(m.key)
        if let b = st?.lastBackend, let r = st?.lastRef,
           let i = m.instances.first(where: { $0.backend == b && ($0.ref == r || (b == .mlx && r.hasPrefix(($0.repoDir ?? "\u{0}") + "/"))) }) {
            return i
        }
        for b in [Backend.ollama, .mlx, .splash, .hf] {
            if let i = m.instances.first(where: { $0.backend == b }) { return i }
        }
        return nil
    }

    /// Favorites, then the last few models used, that are on disk and not already in memory.
    func recentItems() -> [RecentItem] {
        guard let snap = model.library else { return [] }
        let live = liveKeys()
        let byKey = Dictionary(snap.models.map { ($0.key, $0) }, uniquingKeysWith: { a, _ in a })
        func item(_ key: String) -> RecentItem? {
            guard !live.contains(key), let m = byKey[key], let inst = preferredInstance(m) else { return nil }
            let st = store.stats(key)
            return RecentItem(key: key, name: m.name, instance: inst, lastUsed: st?.lastUsed, favorite: st?.favorite ?? false)
        }
        let favs = store.models.filter { $0.value.favorite }
            .sorted { ($0.value.lastUsed ?? .distantPast, $1.key) > ($1.value.lastUsed ?? .distantPast, $0.key) }
            .compactMap { item($0.key) }
        let recents = store.models.filter { !$0.value.favorite && $0.value.lastUsed != nil }
            .sorted { $0.value.lastUsed! > $1.value.lastUsed! }
            .compactMap { item($0.key) }
            .prefix(3)
        return Array((favs + recents).prefix(6))
    }

    // MARK: load

    func loadInstance(_ inst: LibraryInstance) {
        switch inst.backend {
        case .ollama:
            loadModel(inst.ref)
        case .mlx:
            serveInMlx(path: inst.ref, repo: inst.repo ?? shortModelName(inst.ref), server: inst.server ?? "mlx_lm.server")
        case .splash:
            loadSplashModel(inst.ref, languageOnly: inst.languageOnly)
        case .hf:
            serveInSplash(inst.ref)
        }
    }

    /// A Splash model already served by one of the user's LaunchAgents starts through that agent.
    func loadSplashModel(_ id: String, languageOnly: Bool? = nil) {
        if let agent = model.splashServers.first(where: {
            $0.label != nil && $0.label != AppDelegate.panelSplashLabel && $0.agentModel?.lowercased() == id.lowercased()
        }) {
            startSplash(agent)
        } else {
            serveInSplash(id, languageOnly: languageOnly ?? model.splashLanguageOnly.contains(id))
        }
    }

    // MARK: delete

    /// Why a copy can't be deleted right now, or nil.
    func deleteBlocker(_ inst: LibraryInstance) -> String? {
        if let job = model.hfJob, !job.done, !job.failed,
           let repo = inst.repo, job.model.lowercased().contains(repo.lowercased()) {
            return "A download or start for this model is in progress"
        }
        switch inst.backend {
        case .ollama:
            if !model.ollamaUp { return "Ollama isn't running — start it to delete" }
            if model.loaded.contains(where: { $0.name == inst.ref }) { return "Loaded in Ollama — eject it first" }
        case .mlx:
            guard let dir = inst.repoDir else { return nil }
            if let s = model.mlxServers.first(where: { $0.running && ($0.modelPath ?? "").hasPrefix(dir + "/") }) {
                return "Served by MLX on :\(s.port) — stop it first"
            }
            if let s = model.mlxServers.first(where: { $0.configured && !$0.panel && ($0.modelPath ?? "").hasPrefix(dir + "/") }) {
                return "Pinned by “\(s.name)” in mlx_servers.json"
            }
        case .splash:
            let id = inst.ref.lowercased()
            if let s = model.splashServers.first(where: { $0.running && $0.model?.lowercased() == id }) {
                return "Served by Splash on :\(s.port) — stop it first"
            }
            if let s = model.splashServers.first(where: {
                $0.label != nil && $0.label != AppDelegate.panelSplashLabel && $0.agentModel?.lowercased() == id
            }) {
                return "Used by LaunchAgent \(s.label!)"
            }
        case .hf:
            break
        }
        return nil
    }

    /// Delete copies of a model. Ollama deletes through its API (it keeps blobs other tags use);
    /// HF cache files go only when no remaining copy links them.
    func deleteFromLibrary(_ targets: [LibraryInstance], key: String, name: String,
                           completion: @escaping (String?) -> Void) {
        guard let snap = model.library else { completion("Library not scanned yet"); return }
        let freed = LibraryScanner.bytesFreed(targets, in: snap)
        let ids = Set(targets.map(\.id))
        var kept = Set<String>()
        for i in snap.instances where !ids.contains(i.id) { kept.formUnion(i.files.keys) }
        var hubGone = Set<String>()
        for t in targets where t.backend != .ollama { for p in t.files.keys where !kept.contains(p) { hubGone.insert(p) } }
        let allGone = snap.models.first { $0.key == key }.map { $0.instances.allSatisfy { ids.contains($0.id) } } ?? true
        let base = ollamaBase

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            var failure: String?
            for t in targets {
                do {
                    switch t.backend {
                    case .ollama: try AppDelegate.ollamaDelete(t.ref, base: base)
                    case .splash: try LibraryScanner.removeSplashInstall(t)
                    case .mlx, .hf: break
                    }
                } catch {
                    failure = error.localizedDescription
                    break
                }
            }
            if failure == nil {
                do { try LibraryScanner.removeHubBlobs(hubGone) } catch { failure = error.localizedDescription }
            }
            DispatchQueue.main.async {
                guard let self else { return }
                self.store.log(failure == nil ? "delete" : "delete_fail", [
                    "key": key, "backends": targets.map(\.backend.rawValue), "refs": targets.map(\.ref),
                    "freedGB": oneDecimal(freed / 1_073_741_824), "error": failure ?? NSNull(),
                ])
                if failure == nil, allGone { self.store.update(key, name: name) { $0.deletedAt = Date() } }
                if targets.contains(where: { $0.backend == .mlx }) { self.dropPanelMlxEntry(for: targets) }
                self.refreshOllama()
                self.loadMlxCached()
                self.loadSplashInstalled()
                self.rescanLibrary()
                completion(failure)
            }
        }
    }

    static func ollamaDelete(_ name: String, base: String) throws {
        guard let url = URL(string: "\(base)/api/delete") else { return }
        var req = URLRequest(url: url, timeoutInterval: 60)
        req.httpMethod = "DELETE"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["model": name])
        let done = DispatchSemaphore(value: 0)
        var status = 0
        var message = ""
        URLSession.shared.dataTask(with: req) { data, resp, err in
            status = (resp as? HTTPURLResponse)?.statusCode ?? 0
            message = err?.localizedDescription
                ?? data.flatMap { (try? JSONSerialization.jsonObject(with: $0) as? [String: Any])?["error"] as? String } ?? ""
            done.signal()
        }.resume()
        done.wait()
        guard status == 200 else {
            throw LibraryScanner.DeleteError.failed("Ollama couldn't delete \(name)" + (message.isEmpty ? " (HTTP \(status))" : ": \(message)"))
        }
    }

    /// The panel's MLX slot shouldn't point at a model that is gone.
    private func dropPanelMlxEntry(for targets: [LibraryInstance]) {
        let dirs = targets.compactMap(\.repoDir)
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: mlxConfigPath)),
              var arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return }
        let before = arr.count
        arr.removeAll { entry in
            guard entry["panel"] as? Bool == true, let args = entry["args"] as? [String],
                  let i = args.firstIndex(of: "--model"), i + 1 < args.count else { return false }
            return dirs.contains { args[i + 1].hasPrefix($0 + "/") }
        }
        guard arr.count != before,
              let out = try? JSONSerialization.data(withJSONObject: arr, options: [.prettyPrinted, .withoutEscapingSlashes])
        else { return }
        try? out.write(to: URL(fileURLWithPath: mlxConfigPath))
        loadMlxConfig()
        refreshMlx()
    }
}


// MARK: - Content view

// MARK: - UI helpers

private func uiFont(_ size: CGFloat, _ weight: NSFont.Weight = .regular) -> NSFont {
    .systemFont(ofSize: size, weight: weight)
}

private func monoFont(_ size: CGFloat, _ weight: NSFont.Weight = .regular) -> NSFont {
    .monospacedDigitSystemFont(ofSize: size, weight: weight)
}

private func uiLabel(_ text: String, _ size: CGFloat, _ weight: NSFont.Weight = .regular,
                     _ color: NSColor = .labelColor, mono: Bool = false) -> NSTextField {
    let l = NSTextField(labelWithString: text)
    l.font = mono ? monoFont(size, weight) : uiFont(size, weight)
    l.textColor = color
    return l
}

private func trackedLabel(_ text: String, size: CGFloat, weight: NSFont.Weight,
                          color: NSColor, tracking: CGFloat) -> NSTextField {
    let p = NSMutableParagraphStyle()
    p.lineBreakMode = .byTruncatingTail
    return NSTextField(labelWithAttributedString: NSAttributedString(string: text, attributes: [
        .font: uiFont(size, weight), .foregroundColor: color, .kern: tracking, .paragraphStyle: p,
    ]))
}

private func sysSymbol(_ name: String, fallback: String, size: CGFloat,
                       weight: NSFont.Weight = .regular) -> NSImage? {
    let cfg = NSImage.SymbolConfiguration(pointSize: size, weight: weight)
    let base = NSImage(systemSymbolName: name, accessibilityDescription: nil)
        ?? NSImage(systemSymbolName: fallback, accessibilityDescription: nil)
    return base?.withSymbolConfiguration(cfg)
}

/// Infinite gentle opacity pulse for live indicators.
private func pulse(_ layer: CALayer, to alpha: Float = 0.45, period: CFTimeInterval = 1.4) {
    let a = CABasicAnimation(keyPath: "opacity")
    a.fromValue = 1.0
    a.toValue = alpha
    a.duration = period / 2
    a.autoreverses = true
    a.repeatCount = .infinity
    a.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
    layer.add(a, forKey: "pulse")
}

/// Tiny legend glyph: filled dot (pressure) or dashed line (LLM RAM share).
private final class LegendSwatchView: NSView {
    let style: Int // 0 = dot, 1 = dashed line
    var color: NSColor = .systemGreen { didSet { needsDisplay = true } }

    init(style: Int) {
        self.style = style
        super.init(frame: NSRect(x: 0, y: 0, width: 14, height: 10))
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let midY = bounds.midY
        if style == 0 {
            ctx.setFillColor(color.cgColor)
            ctx.fillEllipse(in: CGRect(x: bounds.midX - 3.5, y: midY - 3.5, width: 7, height: 7))
        } else {
            ctx.setStrokeColor(NSColor.controlAccentColor.withAlphaComponent(0.9).cgColor)
            ctx.setLineWidth(1.5)
            ctx.setLineDash(phase: 0, lengths: [3, 2.5])
            ctx.move(to: CGPoint(x: 1, y: midY))
            ctx.addLine(to: CGPoint(x: bounds.width - 1, y: midY))
            ctx.strokePath()
            ctx.setLineDash(phase: 0, lengths: [])
        }
    }
}

// MARK: - Content view

private var reduceMotion: Bool { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion }

/// Section and hero eyebrows share one style so the hierarchy reads the same everywhere.
private func eyebrow(_ text: String) -> NSTextField {
    trackedLabel(text, size: 11, weight: .semibold, color: .secondaryLabelColor, tracking: 0.7)
}

/// Compact recessed button used for every row action, so right edges and widths line up.
private func rowButton(_ title: String, width: CGFloat = 58) -> ClosureButton {
    let btn = ClosureButton(title: title)
    btn.bezelStyle = .recessed
    btn.controlSize = .small
    btn.font = uiFont(10.5, .medium)
    btn.translatesAutoresizingMaskIntoConstraints = false
    btn.widthAnchor.constraint(equalToConstant: width).isActive = true
    return btn
}

/// Pressure text stays readable in light mode: the hue lives on the symbol and fill only.
private func chipFill(_ color: NSColor) -> CGColor { color.withAlphaComponent(0.20).cgColor }

private struct RowAction {
    let title: String
    var enabled = true
    let tooltip: String
    let run: () -> Void
}

private struct ModelRow {
    let dot: NSColor
    var hollow = false                // placeholder rows: ring instead of a filled dot
    var pulse = false                 // transitional states only (starting, ejecting)
    let name: String
    var port: Int? = nil              // always shown as a dim suffix when known
    var size = ""
    var detail = ""
    var action: RowAction? = nil
    var info: String? = nil           // hover text: full ids, pid, context, …
    var dimName = false
}

private struct RowSpec {
    enum Kind {
        case model(ModelRow)
        case load(names: [String], tooltip: String, star: (String) -> Bool, pick: (String) -> Void)
    }
    let key: String
    let kind: Kind

    /// Structural identity. Text-only changes (eject countdown, live sizes) do not
    /// count, so they update labels in place instead of re-animating rows.
    var sig: String {
        switch kind {
        case let .model(r):
            return "\(key)|\(r.pulse ? 1 : 0)|\(r.action?.title ?? "-")|\(r.action?.enabled == true ? 1 : 0)"
        case .load:
            return key
        }
    }
}

/// 34pt row: status dot, name (+ dim port), right-aligned monospaced size · detail, fixed-width action.
/// Rounded 34pt row surface with the shared hover highlight.
private class HoverRowView: NSView {
    private var hoverArea: NSTrackingArea?

    init(height: CGFloat = 34) {
        super.init(frame: NSRect(x: 0, y: 0, width: 10, height: height))
        wantsLayer = true
        layer?.cornerRadius = 10
        layer?.backgroundColor = NSColor.textColor.withAlphaComponent(0.03).cgColor
        translatesAutoresizingMaskIntoConstraints = false
        heightAnchor.constraint(equalToConstant: height).isActive = true
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    // MARK: hover

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let hoverArea { removeTrackingArea(hoverArea) }
        let area = NSTrackingArea(rect: bounds,
                                  options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
                                  owner: self, userInfo: nil)
        addTrackingArea(area)
        hoverArea = area
    }

    override func mouseEntered(with event: NSEvent) { applyHover(true) }
    override func mouseExited(with event: NSEvent) { applyHover(false) }

    private func applyHover(_ on: Bool) {
        guard let layer else { return }
        CATransaction.begin()
        CATransaction.setAnimationDuration(0.15)
        layer.backgroundColor = NSColor.textColor.withAlphaComponent(on ? 0.075 : 0.03).cgColor
        CATransaction.commit()
    }
}

private final class FlippedView: NSView {
    override var isFlipped: Bool { true }
}

private final class MenuAction: NSObject {
    let run: () -> Void
    init(_ run: @escaping () -> Void) { self.run = run }
}

private final class ModelRowView: HoverRowView {
    private let detailLabel: NSTextField

    init(_ r: ModelRow) {
        detailLabel = NSTextField(labelWithString: ModelRowView.detailText(r))
        super.init()

        let row = hstack()
        row.alignment = .centerY
        row.spacing = 8
        row.edgeInsets = NSEdgeInsets(top: 4, left: 10, bottom: 4, right: 10)
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)
        NSLayoutConstraint.activate([
            row.topAnchor.constraint(equalTo: topAnchor),
            row.bottomAnchor.constraint(equalTo: bottomAnchor),
            row.leadingAnchor.constraint(equalTo: leadingAnchor),
            row.trailingAnchor.constraint(equalTo: trailingAnchor),
        ])

        let dot = NSView()
        dot.wantsLayer = true
        dot.layer?.cornerRadius = 4
        if r.hollow {
            dot.layer?.borderWidth = 1.25
            dot.layer?.borderColor = r.dot.cgColor
        } else {
            dot.layer?.backgroundColor = r.dot.cgColor
            dot.layer?.shadowColor = r.dot.cgColor
            dot.layer?.shadowOpacity = 0.55
            dot.layer?.shadowRadius = 2.5
            dot.layer?.shadowOffset = .zero
        }
        dot.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            dot.widthAnchor.constraint(equalToConstant: 8),
            dot.heightAnchor.constraint(equalToConstant: 8),
        ])
        if r.pulse, !reduceMotion, let dotLayer = dot.layer {
            pulse(dotLayer, to: 0.4, period: 1.6)
        }

        let middle = NSMutableParagraphStyle()
        middle.lineBreakMode = .byTruncatingMiddle
        let name = NSAttributedString(string: r.name, attributes: [
            .font: uiFont(13), .paragraphStyle: middle,
            .foregroundColor: r.dimName ? NSColor.secondaryLabelColor : NSColor.labelColor,
        ])
        let nameLabel = NSTextField(labelWithAttributedString: name)
        nameLabel.lineBreakMode = .byTruncatingMiddle
        nameLabel.maximumNumberOfLines = 1
        nameLabel.cell?.truncatesLastVisibleLine = true
        nameLabel.toolTip = r.info ?? r.name
        toolTip = r.info
        nameLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        detailLabel.font = monoFont(10.5)
        detailLabel.textColor = .secondaryLabelColor
        detailLabel.alignment = .right
        detailLabel.lineBreakMode = .byTruncatingTail
        detailLabel.maximumNumberOfLines = 1
        detailLabel.setContentHuggingPriority(.required, for: .horizontal)
        detailLabel.setContentCompressionResistancePriority(.defaultHigh, for: .horizontal)

        row.addView(dot, in: .leading)
        row.addView(nameLabel, in: .leading)
        if let port = r.port {
            // own label so middle-truncating a long name never eats the port
            let portLabel = uiLabel(":\(port)", 11, .regular, .secondaryLabelColor, mono: true)
            portLabel.setContentCompressionResistancePriority(.required, for: .horizontal)
            portLabel.setContentHuggingPriority(.required, for: .horizontal)
            row.addView(portLabel, in: .leading)
            row.setCustomSpacing(4, after: nameLabel)
        }
        row.addView(detailLabel, in: .trailing)

        if let a = r.action {
            let btn = rowButton(a.title)
            btn.isEnabled = a.enabled
            btn.toolTip = a.tooltip
            btn.onAction = a.run
            row.addView(btn, in: .trailing)
        }
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    static func detailText(_ r: ModelRow) -> String {
        [r.size, r.detail].filter { !$0.isEmpty }.joined(separator: " · ")
    }

    func update(_ r: ModelRow) {
        detailLabel.stringValue = ModelRowView.detailText(r)
        toolTip = r.info
    }

}

final class PanelContentView: NSView, NSSearchFieldDelegate, NSMenuDelegate {
    private let model: PanelModel
    private weak var appDelegate: AppDelegate?

    private var graphView: PressureGraphView!
    private var statsLabel: NSTextField!
    private var heroValueLabel: NSTextField!
    private var heroTotalLabel: NSTextField!
    private var pressureChip: NSStackView!
    private var pressureChipLabel: NSTextField!
    private var pressureChipSymbol: NSImageView!
    private var swapChip: NSStackView!
    private var swapChipLabel: NSTextField!
    private var ollamaMetaLabel: NSTextField!
    private var ejectAllButton: ClosureButton!
    private var mlxMetaLabel: NSTextField!
    private var splashMetaLabel: NSTextField!
    private var ollamaRows: NSStackView!
    private var mlxRows: NSStackView!
    private var splashRows: NSStackView!
    /// Hairline, header and rows of each backend section, hidden when that backend isn't installed.
    private var sectionViews: [String: [NSView]] = [:]
    private var rootStack: NSStackView!
    /// Popover height that fits the current rows (16 top + 12 bottom around the root stack).
    private(set) var fittedHeight: CGFloat = 600

    // Hugging Face
    private var hfMetaLabel: NSTextField!
    private var hfBody: NSStackView!
    private var hfSearch: NSSearchField!
    /// Debug screenshots: show the query a `--hf-search` run searched for.
    func showSearchQuery(_ query: String) { hfSearch.stringValue = query }
    private var hfRecent: NSButton!
    private var hfStatusLabel: NSTextField!
    private var hfRepoRow: NSView!
    private var hfRepoPopup: NSPopUpButton!
    private var hfQuantRow: NSView!
    private var hfQuantPopup: NSPopUpButton!
    private var hfInfoLabel: NSTextField!
    private var hfActionRow: NSView!
    private var hfOllamaButton: ClosureButton!
    private var hfSplashButton: ClosureButton!
    private var hfMlxButton: ClosureButton!
    private var hfFormatControl: NSSegmentedControl!
    private var hfJobRow: NSView!
    private var hfProgress: NSProgressIndicator!
    private var hfJobLabel: NSTextField!
    private var hfJobButton: ClosureButton!
    private var hfResultsSig = ""
    private var hfDetailSig = ""

    private var refreshTimer: Timer?
    private var rowCache: [String: NSView] = [:]
    private var lastRowSig = ""

    // Tabs
    private enum Tab: Int { case running, library, get }
    private var tabControl: NSSegmentedControl!
    private var tabStacks: [NSStackView] = []
    private var tab: Tab {
        get { Tab(rawValue: UserDefaults.standard.integer(forKey: "panelTab")) ?? .running }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "panelTab") }
    }

    // Recent strip (Running tab)
    private var recentSection: NSStackView!
    private var recentRows: NSStackView!

    // Library tab
    private struct PendingDelete {
        let key: String
        let ids: [String]
        var busy = false
        var error: String?
    }
    private var libDiskLabel: NSTextField!
    private var libDupLabel: NSTextField!
    private var libSearch: NSSearchField!
    private var libFilter: NSSegmentedControl!
    private var libScroll: NSScrollView!
    private var libScrollHeight: NSLayoutConstraint!
    private var libList: NSStackView!
    private var libSig = ""
    private var pendingDelete: PendingDelete?

    init(model: PanelModel, appDelegate: AppDelegate) {
        self.model = model
        self.appDelegate = appDelegate
        super.init(frame: NSRect(x: 0, y: 0, width: 400, height: 600))
        buildUI()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            // Memory sampling stays on the app-delegate timer (one sample per 2 s,
            // a true 5-minute window); this tick only refreshes process state, and
            // only while the panel is on screen (the delegate keeps a slower cadence).
            guard let w = self.window, w.isVisible, w.occlusionState.contains(.visible) else { return }
            self.appDelegate?.refreshOllama()
            self.appDelegate?.refreshMlx()
            self.appDelegate?.refreshSplash()
            if self.tab == .library, let snap = self.model.library, Date().timeIntervalSince(snap.scannedAt) > 60 {
                self.appDelegate?.rescanLibrary()
            }
            self.rebuild()
        }
        applyTab()
        rebuild()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    // MARK: build

    private func hairline() -> NSView {
        let v = NSView()
        v.wantsLayer = true
        v.layer?.backgroundColor = NSColor.separatorColor.withAlphaComponent(0.6).cgColor
        v.translatesAutoresizingMaskIntoConstraints = false
        v.heightAnchor.constraint(equalToConstant: 0.5).isActive = true
        return v
    }

    private func makeSectionHeader(_ title: String, accessory: NSView? = nil) -> (NSStackView, NSTextField) {
        let h = hstack()
        h.alignment = .centerY
        h.spacing = 8
        h.addArrangedSubview(eyebrow(title))
        let meta = uiLabel("", 10.5, .regular, .secondaryLabelColor, mono: true)
        h.addView(meta, in: .trailing)
        if let accessory { h.addView(accessory, in: .trailing) }
        return (h, meta)
    }

    private func makeLegendChip(swatch: LegendSwatchView, text: String) -> NSView {
        let chip = hstack()
        chip.alignment = .centerY
        chip.spacing = 5
        chip.edgeInsets = NSEdgeInsets(top: 3, left: 8, bottom: 3, right: 8)
        chip.wantsLayer = true
        chip.layer?.cornerRadius = 9
        chip.layer?.backgroundColor = NSColor.textColor.withAlphaComponent(0.05).cgColor
        chip.addArrangedSubview(swatch)
        chip.addArrangedSubview(uiLabel(text, 10.5, .medium, .secondaryLabelColor))
        chip.setHuggingPriority(.required, for: .horizontal)
        return chip
    }

    /// Adds `view` full-width to the root stack with `after` points below it.
    private func addFull(_ view: NSView, to root: NSStackView, after spacing: CGFloat) {
        root.addArrangedSubview(view)
        view.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        root.setCustomSpacing(spacing, after: view)
    }

    private func makeRowStack() -> NSStackView {
        let s = NSStackView()
        s.orientation = .vertical
        s.alignment = .width
        s.spacing = 0
        s.detachesHiddenViews = true
        return s
    }

    private func buildUI() {
        let glass = NSGlassEffectView()
        glass.cornerRadius = 24
        glass.translatesAutoresizingMaskIntoConstraints = false
        addSubview(glass)

        let root = NSStackView()
        rootStack = root
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 8
        root.detachesHiddenViews = true
        root.translatesAutoresizingMaskIntoConstraints = false
        glass.addSubview(root)

        NSLayoutConstraint.activate([
            glass.topAnchor.constraint(equalTo: topAnchor),
            glass.bottomAnchor.constraint(equalTo: bottomAnchor),
            glass.leadingAnchor.constraint(equalTo: leadingAnchor),
            glass.trailingAnchor.constraint(equalTo: trailingAnchor),
            root.topAnchor.constraint(equalTo: glass.topAnchor, constant: 16),
            root.leadingAnchor.constraint(equalTo: glass.leadingAnchor, constant: 16),
            root.trailingAnchor.constraint(equalTo: glass.trailingAnchor, constant: -16),
            root.bottomAnchor.constraint(lessThanOrEqualTo: glass.bottomAnchor, constant: -12),
        ])

        // header — title, refresh (⌘R), more menu (Activity Monitor, Quit)
        let header = hstack()
        header.spacing = 7
        header.alignment = .centerY
        let icon = NSImageView(image: sysSymbol("memorychip", fallback: "memorychip", size: 13, weight: .medium) ?? NSImage())
        icon.contentTintColor = .secondaryLabelColor
        icon.symbolConfiguration = .preferringHierarchical()
        header.addArrangedSubview(icon)
        header.addArrangedSubview(uiLabel("Local AI Manager", 15, .semibold))
        let refresh = ClosureButton(title: "")
        refresh.bezelStyle = .recessed
        refresh.controlSize = .small
        refresh.image = sysSymbol("arrow.clockwise", fallback: "arrow.clockwise.circle", size: 11)
        refresh.imagePosition = .imageOnly
        refresh.toolTip = "Refresh now (⌘R)"
        refresh.keyEquivalent = "r"
        refresh.keyEquivalentModifierMask = [.command]
        refresh.onAction = { [weak self] in self?.refreshNow() }
        header.addView(refresh, in: .trailing)
        header.addView(makeMoreMenu(), in: .trailing)
        addFull(header, to: root, after: 10)

        // tabs — Running (memory now) · Library (on disk) · Get (Hugging Face)
        tabControl = NSSegmentedControl(labels: ["Running", "Library", "Get"], trackingMode: .selectOne,
                                        target: self, action: #selector(tabChanged))
        tabControl.segmentDistribution = .fillEqually
        tabControl.controlSize = .small
        tabControl.font = uiFont(11, .medium)
        tabControl.selectedSegment = tab.rawValue
        tabControl.setToolTip("What's in memory now", forSegment: 0)
        tabControl.setToolTip("Every model on disk — favorites, history, delete", forSegment: 1)
        tabControl.setToolTip("Find models on Hugging Face", forSegment: 2)
        addFull(tabControl, to: root, after: 12)
        for _ in 0..<3 {
            let t = NSStackView()
            t.orientation = .vertical
            t.alignment = .leading
            t.spacing = 8
            t.detachesHiddenViews = true
            tabStacks.append(t)
            addFull(t, to: root, after: 4)
        }
        let run = tabStacks[0]

        // hero — total LLM RAM of physical, pressure chip (+ swap warning when heavy)
        let hero = hstack()
        hero.alignment = .top
        let heroLeft = NSStackView()
        heroLeft.orientation = .vertical
        heroLeft.alignment = .leading
        heroLeft.spacing = 1
        heroLeft.addArrangedSubview(eyebrow("TOTAL LLM RAM"))
        let figureRow = hstack()
        figureRow.alignment = .firstBaseline
        figureRow.spacing = 4
        heroValueLabel = uiLabel("0.0", 26, .semibold, mono: true)
        heroTotalLabel = uiLabel(String(format: "GB of %.0f GB", model.totalGB), 12, .medium, .secondaryLabelColor)
        figureRow.addArrangedSubview(heroValueLabel)
        figureRow.addArrangedSubview(heroTotalLabel)
        heroLeft.addArrangedSubview(figureRow)

        let heroRight = NSStackView()
        heroRight.orientation = .vertical
        heroRight.alignment = .trailing
        heroRight.spacing = 6
        heroRight.detachesHiddenViews = true

        pressureChip = hstack()
        pressureChip.alignment = .centerY
        pressureChip.spacing = 5
        pressureChip.edgeInsets = NSEdgeInsets(top: 4, left: 9, bottom: 4, right: 9)
        pressureChip.wantsLayer = true
        pressureChip.layer?.cornerRadius = 11
        pressureChip.toolTip = "macOS memory pressure"
        pressureChipSymbol = NSImageView(image: NSImage())
        pressureChipLabel = uiLabel("Normal", 11, .semibold, .labelColor)
        pressureChip.addArrangedSubview(pressureChipSymbol)
        pressureChip.addArrangedSubview(pressureChipLabel)
        pressureChip.setHuggingPriority(.required, for: .horizontal)
        heroRight.addArrangedSubview(pressureChip)

        swapChip = hstack()
        swapChip.alignment = .centerY
        swapChip.spacing = 5
        swapChip.edgeInsets = NSEdgeInsets(top: 4, left: 9, bottom: 4, right: 9)
        swapChip.wantsLayer = true
        swapChip.layer?.cornerRadius = 11
        swapChip.toolTip = "Swap usage above 16 GB — the system is under heavy memory pressure"
        let swapSymbol = NSImageView(image: sysSymbol("externaldrive.badge.exclamationmark",
                                                      fallback: "externaldrive", size: 10, weight: .medium) ?? NSImage())
        swapSymbol.contentTintColor = .systemOrange
        swapChipLabel = uiLabel("", 11, .semibold, .labelColor)
        swapChip.addArrangedSubview(swapSymbol)
        swapChip.addArrangedSubview(swapChipLabel)
        swapChip.isHidden = true
        swapChip.setHuggingPriority(.required, for: .horizontal)
        heroRight.addArrangedSubview(swapChip)

        hero.addView(heroLeft, in: .leading)
        hero.addView(heroRight, in: .trailing)
        addFull(hero, to: run, after: 10)

        // graph card (rounded 12, recessed surface + hairline border)
        let card = NSView()
        card.wantsLayer = true
        card.layer?.backgroundColor = NSColor.textColor.withAlphaComponent(0.045).cgColor
        card.layer?.cornerRadius = 12
        card.layer?.borderWidth = 0.5
        card.layer?.borderColor = NSColor.textColor.withAlphaComponent(0.09).cgColor
        card.toolTip = "Last 5 minutes · Solid: RAM used (app + wired + compressed) · Dashed: LLM share"
        graphView = PressureGraphView()
        graphView.model = model
        graphView.translatesAutoresizingMaskIntoConstraints = false
        card.addSubview(graphView)
        NSLayoutConstraint.activate([
            graphView.topAnchor.constraint(equalTo: card.topAnchor, constant: 6),
            graphView.bottomAnchor.constraint(equalTo: card.bottomAnchor, constant: -6),
            graphView.leadingAnchor.constraint(equalTo: card.leadingAnchor),
            graphView.trailingAnchor.constraint(equalTo: card.trailingAnchor),
            graphView.heightAnchor.constraint(equalToConstant: 76),
        ])
        addFull(card, to: run, after: 6)

        // legend chips + stats caption
        let usedSwatch = LegendSwatchView(style: 0)
        usedSwatch.color = PressureGraphView.usedColor
        let llmSwatch = LegendSwatchView(style: 1)
        for s in [usedSwatch, llmSwatch] {
            s.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                s.widthAnchor.constraint(equalToConstant: 14),
                s.heightAnchor.constraint(equalToConstant: 10),
            ])
        }
        let legend = hstack()
        legend.spacing = 8
        legend.setHuggingPriority(.required, for: .horizontal)
        legend.addArrangedSubview(makeLegendChip(swatch: usedSwatch, text: "RAM used"))
        legend.addArrangedSubview(makeLegendChip(swatch: llmSwatch, text: "LLM share"))
        run.addArrangedSubview(legend)
        run.setCustomSpacing(6, after: legend)

        statsLabel = uiLabel(" ", 10.5, .regular, .secondaryLabelColor, mono: true)
        statsLabel.lineBreakMode = .byTruncatingTail
        statsLabel.maximumNumberOfLines = 1
        addFull(statsLabel, to: run, after: 10)

        // recent — favorites and last used, one click to load again
        recentSection = NSStackView()
        recentSection.orientation = .vertical
        recentSection.alignment = .leading
        recentSection.spacing = 6
        let (recentHeader, recentMeta) = makeSectionHeader("RECENT")
        recentMeta.stringValue = "favorites · last used"
        addFull(recentSection, to: run, after: 10)
        addFull(hairline(), to: recentSection, after: 10)
        addFull(recentHeader, to: recentSection, after: 6)
        recentRows = makeRowStack()
        addFull(recentRows, to: recentSection, after: 0)

        // backend sections
        ejectAllButton = ClosureButton(title: "Eject All")
        ejectAllButton.bezelStyle = .inline
        ejectAllButton.isBordered = false
        ejectAllButton.controlSize = .small
        ejectAllButton.font = uiFont(10.5, .medium)
        ejectAllButton.contentTintColor = .controlAccentColor
        ejectAllButton.toolTip = "Unload every Ollama model from memory"
        ejectAllButton.onAction = { [weak self] in
            self?.appDelegate?.flushAll()
            self?.rebuild()
        }

        for (title, accessory) in [("OLLAMA", ejectAllButton as NSView?), ("MLX SERVERS", nil), ("SPLASH", nil)] {
            let line = hairline()
            addFull(line, to: run, after: 10)
            let (header, meta) = makeSectionHeader(title, accessory: accessory)
            addFull(header, to: run, after: 6)
            let rows = makeRowStack()
            addFull(rows, to: run, after: 10)
            sectionViews[title] = [line, header, rows]
            switch title {
            case "OLLAMA": ollamaMetaLabel = meta; ollamaRows = rows
            case "MLX SERVERS": mlxMetaLabel = meta; mlxRows = rows
            default: splashMetaLabel = meta; splashRows = rows
            }
        }

        buildLibrary(in: tabStacks[1])
        buildHFSection(in: tabStacks[2])

        // footer — keyboard hint
        addFull(hairline(), to: root, after: 8)
        let hint = NSTextField(labelWithAttributedString: NSAttributedString(
            string: "⌘R  Refresh · Updates every 2 s", attributes: [
                .font: uiFont(10),
                .foregroundColor: NSColor.tertiaryLabelColor,
                .kern: 0.3,
                .paragraphStyle: {
                    let p = NSMutableParagraphStyle()
                    p.alignment = .center
                    return p
                }(),
            ]))
        root.addArrangedSubview(hint)
        hint.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
    }

    private func makeMoreMenu() -> NSView {
        let more = NSPopUpButton(frame: .zero, pullsDown: true)
        more.bezelStyle = .recessed
        more.controlSize = .small
        more.isBordered = false
        (more.cell as? NSPopUpButtonCell)?.arrowPosition = .noArrow
        let title = NSMenuItem(title: "", action: nil, keyEquivalent: "")
        title.image = sysSymbol("ellipsis.circle", fallback: "ellipsis", size: 12)
        more.menu?.addItem(title)
        let items: [(String, Selector, String)] = [
            ("Refresh", #selector(menuRefresh), "r"),
            ("Open Activity Monitor", #selector(menuActivityMonitor), ""),
        ]
        for (t, sel, key) in items {
            let item = NSMenuItem(title: t, action: sel, keyEquivalent: key)
            item.target = self
            more.menu?.addItem(item)
        }
        // shown only when a Splish checkout is found (menuNeedsUpdate)
        more.menu?.addItem(.separator())
        for engine in ["Splash", "Splish"] {
            let item = NSMenuItem(title: "Run Splash models with \(engine)", action: #selector(menuSplashEngine(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = engine
            item.tag = 7
            more.menu?.addItem(item)
        }
        more.menu?.delegate = self
        more.menu?.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Local AI Manager", action: #selector(menuQuit), keyEquivalent: "q")
        quit.target = self
        more.menu?.addItem(quit)
        more.toolTip = "More"
        return more
    }

    @objc private func menuRefresh() { refreshNow() }
    @objc private func menuSplashEngine(_ sender: NSMenuItem) {
        UserDefaults.standard.set(sender.representedObject as? String, forKey: "splashEngine")
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        let hasSplish = AppDelegate.splishRoot != nil
        let current = appDelegate?.preferredSplashEngine
        for item in menu.items where item.tag == 7 {
            item.isHidden = !hasSplish
            item.state = (item.representedObject as? String) == current ? .on : .off
        }
        // the separator before the engine items goes with them
        if let i = menu.items.firstIndex(where: { $0.tag == 7 }), i > 0 { menu.items[i - 1].isHidden = !hasSplish }
    }
    @objc private func menuQuit() { NSApp.terminate(nil) }
    @objc private func menuActivityMonitor() {
        NSWorkspace.shared.openApplication(
            at: URL(fileURLWithPath: "/System/Applications/Utilities/Activity Monitor.app"),
            configuration: NSWorkspace.OpenConfiguration())
    }

    private func refreshNow() {
        appDelegate?.refreshAll()
        rebuild()
    }

    // MARK: Hugging Face section

    private func formLabel(_ text: String) -> NSTextField {
        let l = uiLabel(text, 11, .medium, .secondaryLabelColor)
        l.translatesAutoresizingMaskIntoConstraints = false
        l.widthAnchor.constraint(equalToConstant: 40).isActive = true
        return l
    }

    private func formRow(_ label: String, _ control: NSView) -> NSView {
        let row = hstack()
        row.alignment = .centerY
        row.spacing = 6
        row.edgeInsets = NSEdgeInsets(top: 0, left: 10, bottom: 0, right: 10)
        row.distribution = .fill
        row.addArrangedSubview(formLabel(label))
        row.addArrangedSubview(control)
        control.setContentHuggingPriority(.defaultLow, for: .horizontal)
        control.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return row
    }

    private func buildHFSection(in root: NSStackView) {
        let (header, meta) = makeSectionHeader("HUGGING FACE")
        hfMetaLabel = meta
        addFull(header, to: root, after: 8)

        hfBody = NSStackView()
        hfBody.orientation = .vertical
        hfBody.alignment = .width
        hfBody.spacing = 6
        hfBody.detachesHiddenViews = true

        // search + recent filter
        hfSearch = NSSearchField()
        hfSearch.controlSize = .small
        hfSearch.font = uiFont(12)
        hfSearch.placeholderString = "Search GGUF models"
        hfSearch.sendsWholeSearchString = true
        hfSearch.sendsSearchStringImmediately = false
        hfSearch.target = self
        hfSearch.action = #selector(hfSearchSubmitted)
        hfSearch.setContentHuggingPriority(.defaultLow, for: .horizontal)
        hfRecent = NSButton(checkboxWithTitle: "Last 90 days", target: self, action: #selector(hfSearchSubmitted))
        hfRecent.controlSize = .small
        hfRecent.font = uiFont(11)
        hfRecent.state = .on
        hfRecent.toolTip = "Only repos created in the last 90 days"
        hfFormatControl = NSSegmentedControl(labels: [HFFormat.gguf.rawValue, HFFormat.mlx.rawValue],
                                             trackingMode: .selectOne, target: self, action: #selector(hfFormatChanged))
        hfFormatControl.controlSize = .small
        hfFormatControl.font = uiFont(10.5, .medium)
        hfFormatControl.selectedSegment = model.hfFormat == .gguf ? 0 : 1
        hfFormatControl.setToolTip("GGUF — for Ollama and Splash", forSegment: 0)
        hfFormatControl.setToolTip("MLX — served by mlx-lm / mlx-vlm", forSegment: 1)
        hfFormatControl.setContentHuggingPriority(.required, for: .horizontal)
        let searchRow = hstack()
        searchRow.alignment = .centerY
        searchRow.spacing = 8
        searchRow.edgeInsets = NSEdgeInsets(top: 0, left: 0, bottom: 0, right: 0)
        searchRow.addArrangedSubview(hfSearch)
        searchRow.addArrangedSubview(hfFormatControl)
        hfBody.addArrangedSubview(searchRow)

        hfStatusLabel = uiLabel("", 10.5, .regular, .secondaryLabelColor)
        hfStatusLabel.lineBreakMode = .byTruncatingTail
        hfStatusLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        hfRecent.setContentHuggingPriority(.required, for: .horizontal)
        let statusWrap = hstack()
        statusWrap.alignment = .centerY
        statusWrap.edgeInsets = NSEdgeInsets(top: 0, left: 2, bottom: 0, right: 0)
        statusWrap.addView(hfStatusLabel, in: .leading)
        statusWrap.addView(hfRecent, in: .trailing)
        hfBody.addArrangedSubview(statusWrap)

        hfRepoPopup = NSPopUpButton()
        hfRepoPopup.controlSize = .small
        hfRepoPopup.font = uiFont(11)
        (hfRepoPopup.cell as? NSPopUpButtonCell)?.lineBreakMode = .byTruncatingMiddle
        hfRepoPopup.target = self
        hfRepoPopup.action = #selector(hfRepoPicked)
        hfRepoRow = formRow("Model", hfRepoPopup)
        hfBody.addArrangedSubview(hfRepoRow)

        hfQuantPopup = NSPopUpButton()
        hfQuantPopup.controlSize = .small
        hfQuantPopup.font = monoFont(11)
        hfQuantPopup.target = self
        hfQuantPopup.action = #selector(hfQuantPicked)
        hfQuantRow = formRow("Quant", hfQuantPopup)
        hfBody.addArrangedSubview(hfQuantRow)

        // info + targets
        hfInfoLabel = uiLabel("", 10.5, .regular, .secondaryLabelColor, mono: true)
        hfInfoLabel.lineBreakMode = .byTruncatingTail
        hfInfoLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        hfOllamaButton = rowButton("Ollama", width: 72)
        hfOllamaButton.image = sysSymbol("arrow.down.circle", fallback: "arrow.down", size: 10)
        hfOllamaButton.imagePosition = .imageLeading
        hfOllamaButton.onAction = { [weak self] in self?.hfSend(to: .ollama) }
        hfSplashButton = rowButton("Splash", width: 72)
        hfSplashButton.image = sysSymbol("play.circle", fallback: "play", size: 10)
        hfSplashButton.imagePosition = .imageLeading
        hfSplashButton.onAction = { [weak self] in self?.hfSend(to: .splash) }
        hfMlxButton = rowButton("MLX", width: 72)
        hfMlxButton.image = sysSymbol("arrow.down.circle", fallback: "arrow.down", size: 10)
        hfMlxButton.imagePosition = .imageLeading
        hfMlxButton.onAction = { [weak self] in self?.hfSend(to: .mlx) }
        let actionRow = hstack()
        actionRow.alignment = .centerY
        actionRow.spacing = 6
        actionRow.edgeInsets = NSEdgeInsets(top: 2, left: 10, bottom: 0, right: 10)
        actionRow.addView(hfInfoLabel, in: .leading)
        actionRow.addView(hfOllamaButton, in: .trailing)
        actionRow.addView(hfSplashButton, in: .trailing)
        actionRow.addView(hfMlxButton, in: .trailing)
        hfActionRow = actionRow
        hfBody.addArrangedSubview(actionRow)

        // job progress
        hfProgress = NSProgressIndicator()
        hfProgress.style = .bar
        hfProgress.controlSize = .small
        hfProgress.minValue = 0
        hfProgress.maxValue = 1
        hfProgress.translatesAutoresizingMaskIntoConstraints = false
        hfProgress.widthAnchor.constraint(equalToConstant: 64).isActive = true
        hfJobLabel = uiLabel("", 10.5, .regular, .secondaryLabelColor, mono: true)
        hfJobLabel.lineBreakMode = .byTruncatingMiddle
        hfJobLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        hfJobButton = rowButton("Cancel")
        hfJobButton.onAction = { [weak self] in self?.appDelegate?.cancelHFJob() }
        let jobRow = hstack()
        jobRow.alignment = .centerY
        jobRow.spacing = 8
        jobRow.edgeInsets = NSEdgeInsets(top: 0, left: 10, bottom: 0, right: 10)
        jobRow.addView(hfProgress, in: .leading)
        jobRow.addView(hfJobLabel, in: .leading)
        jobRow.addView(hfJobButton, in: .trailing)
        hfJobRow = jobRow
        hfBody.addArrangedSubview(jobRow)

        for v in hfBody.arrangedSubviews {
            v.widthAnchor.constraint(equalTo: hfBody.widthAnchor).isActive = true
        }
        addFull(hfBody, to: root, after: 10)
    }

    @objc private func hfSearchSubmitted() {
        let q = hfSearch.stringValue
        guard !q.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        appDelegate?.searchHF(q, recentOnly: hfRecent.state == .on)
    }

    @objc private func hfFormatChanged() {
        model.hfFormat = hfFormatControl.selectedSegment == 1 ? .mlx : .gguf
        model.hfResults = []
        model.hfDetail = nil
        model.hfError = nil
        model.hfSearching = false
        rebuild()
        hfSearchSubmitted()
    }

    @objc private func hfRepoPicked() {
        guard let id = hfRepoPopup.selectedItem?.representedObject as? String else { return }
        appDelegate?.fetchHFDetail(id)
    }

    @objc private func hfQuantPicked() { updateHF() }

    private var selectedQuant: HFQuant? {
        guard let detail = model.hfDetail, let tag = hfQuantPopup.selectedItem?.representedObject as? String
        else { return nil }
        return detail.quants.first { $0.tag == tag }
    }

    private func hfSend(to target: HFJob.Target) {
        guard let detail = model.hfDetail, let q = selectedQuant else { return }
        switch target {
        case .ollama: appDelegate?.pullToOllama(repo: detail.id, tag: q.tag)
        case .splash: appDelegate?.serveInSplash("\(detail.id):\(q.tag)")
        case .mlx:
            guard let server = appDelegate?.mlxServer(for: detail.architecture, vision: detail.vision) else { return }
            appDelegate?.downloadToMlx(repo: detail.id, sizeGB: q.sizeGB, server: server)
        }
    }

    /// Everything here is updated in place: rebuilding would drop focus and selections.
    private func updateHF() {

        let job = model.hfJob
        let jobActive = job.map { !$0.done && !$0.failed } ?? false
        if let job, jobActive {
            hfMetaLabel.stringValue = job.fraction.map { "\(job.target.rawValue) \(Int($0 * 100))%" }
                ?? "\(job.target.rawValue)…"
        } else {
            hfMetaLabel.stringValue = ""
        }
        hfMetaLabel.textColor = .secondaryLabelColor

        // results → model popup (rebuilt only when the result list changes)
        let resultsSig = model.hfResults.map(\.id).joined(separator: "|")
        if resultsSig != hfResultsSig {
            hfResultsSig = resultsSig
            hfRepoPopup.removeAllItems()
            let rel = RelativeDateTimeFormatter()
            rel.unitsStyle = .short
            for r in model.hfResults {
                hfRepoPopup.addItem(withTitle: "\(r.id)  ·  \(AppDelegate.formatCount(r.downloads)) ↓")
                hfRepoPopup.lastItem?.representedObject = r.id
                hfRepoPopup.lastItem?.toolTip = "\(r.id)\n\(r.downloads) downloads · \(r.likes) likes"
                    + (r.createdAt.map { "\ncreated \(rel.localizedString(for: $0, relativeTo: Date()))" } ?? "")
            }
        }
        hfRepoRow.isHidden = model.hfResults.isEmpty

        // detail → quant popup, defaulting to a Q4_K_M-class quant
        let detailSig = model.hfDetail.map { d in d.id + d.quants.map(\.tag).joined() } ?? ""
        if detailSig != hfDetailSig {
            hfDetailSig = detailSig
            hfQuantPopup.removeAllItems()
            for q in model.hfDetail?.quants ?? [] {
                hfQuantPopup.addItem(withTitle: String(format: "%@  ·  %.1f GB", q.tag, q.sizeGB))
                hfQuantPopup.lastItem?.representedObject = q.tag
            }
            let tags = model.hfDetail?.quants.map(\.tag) ?? []
            if let i = tags.firstIndex(where: { $0.uppercased().hasSuffix("Q4_K_M") })
                ?? tags.firstIndex(where: { $0.uppercased().contains("Q4") }) {
                hfQuantPopup.selectItem(at: i)
            }
        }
        hfQuantRow.isHidden = (model.hfDetail?.quants.isEmpty ?? true)

        // status line: searching / errors / loading detail
        var status = ""
        if model.hfSearching { status = "Searching Hugging Face…" }
        else if let err = model.hfError { status = err }
        else if !model.hfResults.isEmpty, model.hfDetail == nil { status = "Reading repo…" }
        else if model.hfResults.isEmpty {
            status = model.hfFormat == .gguf ? "For Ollama and Splash" : "Served on :\(AppDelegate.panelMlxPort) by mlx-lm"
        }
        hfStatusLabel.stringValue = status
        hfStatusLabel.textColor = model.hfError != nil ? .systemOrange : .secondaryLabelColor
        hfSearch.placeholderString = "Search \(model.hfFormat.rawValue) models"

        // info + target buttons
        let q = selectedQuant
        hfActionRow.isHidden = q == nil
        if let detail = model.hfDetail, let q {
            var parts: [String] = []
            if let arch = detail.architecture { parts.append(arch) }
            if let ctx = detail.contextTokens { parts.append("\(AppDelegate.formatTokens(ctx)) ctx") }
            let budget = model.totalGB * 0.75
            var color = NSColor.secondaryLabelColor
            if q.sizeGB > budget {
                parts.append(String(format: "too large for %.0f GB", model.totalGB))
                color = .systemOrange
            } else if q.sizeGB + model.llmGB > budget {
                parts.append("eject others first")
                color = .systemOrange
            } else {
                parts.append("fits")
            }
            hfInfoLabel.stringValue = parts.joined(separator: " · ")
            hfInfoLabel.textColor = color
            hfInfoLabel.toolTip = String(format: "%@:%@ · %.1f GB weights\nBudget ≈ %.0f GB (75%% of RAM); LLMs now hold %.1f GB",
                                         detail.id, q.tag, q.sizeGB, budget, model.llmGB)

            let isMlx = detail.format == .mlx
            hfOllamaButton.isHidden = isMlx
            hfSplashButton.isHidden = isMlx
            hfMlxButton.isHidden = !isMlx
            let server = isMlx ? appDelegate?.mlxServer(for: detail.architecture, vision: detail.vision) : nil
            hfMlxButton.isEnabled = server != nil && !jobActive
            hfMlxButton.toolTip = server == nil
                ? "The MLX venv can't run model type \(detail.architecture ?? "?") — update mlx-lm / mlx-vlm"
                : "Download \(detail.id) (\(String(format: "%.1f GB", q.sizeGB))) and serve it with \(server!) on :\(AppDelegate.panelMlxPort) — replaces the model running there"

            hfOllamaButton.isEnabled = model.ollamaUp && !jobActive
            hfOllamaButton.toolTip = model.ollamaUp
                ? "Download hf.co/\(detail.id):\(q.tag) into Ollama, then load it"
                : "Ollama isn't running"
            let arch = detail.architecture
            let splashOK = arch.map { AppDelegate.splashArchitectures.contains($0) } ?? true
            hfSplashButton.isEnabled = splashOK && !jobActive
            hfSplashButton.toolTip = !splashOK
                ? "Splash runs Qwen3.5-family models only (this one is \(arch ?? "?"))"
                : "Serve \(detail.id):\(q.tag) on :\(AppDelegate.panelSplashPort) — replaces the model running there"
                    + (arch == nil ? "\nArchitecture unknown — Splash may reject it" : "")
        }

        // job line
        hfJobRow.isHidden = job == nil
        if let job {
            if let f = job.fraction {
                hfProgress.isIndeterminate = false
                hfProgress.stopAnimation(nil)
                hfProgress.doubleValue = f
            } else {
                hfProgress.isIndeterminate = true
                if jobActive { hfProgress.startAnimation(nil) } else { hfProgress.stopAnimation(nil) }
            }
            hfProgress.isHidden = !jobActive && job.fraction == nil
            hfJobLabel.stringValue = "\(job.target.rawValue): \(job.status)"
            hfJobLabel.toolTip = "\(job.model)\n\(job.status)"
            hfJobLabel.textColor = job.failed ? .systemOrange : .secondaryLabelColor
            hfJobButton.title = jobActive ? "Cancel" : "Clear"
            hfJobButton.toolTip = !jobActive ? "Dismiss"
                : job.target == .splash ? "Stop the Splash server that is starting"
                : job.target == .mlx && job.started != nil ? "Stop the MLX server that is starting"
                : "Stop the download (it resumes next time)"
        }
    }

    // MARK: refresh

    private var pressureSymbolName: String {
        switch model.pressureLevel {
        case 1: return "gauge.with.dots.needle.33percent"
        case 2: return "gauge.with.dots.needle.67percent"
        default: return "gauge.with.dots.needle.100percent"
        }
    }

    func rebuild() {
        heroValueLabel.stringValue = String(format: "%.1f", model.llmGB)

        let pc = model.pressureColor
        pressureChip.layer?.backgroundColor = chipFill(pc)
        pressureChipLabel.stringValue = model.pressureText
        pressureChipSymbol.contentTintColor = pc
        pressureChipSymbol.image = sysSymbol(pressureSymbolName, fallback: "gauge", size: 11, weight: .semibold)

        if model.swapGB > 16 {
            swapChip.isHidden = false
            swapChip.layer?.backgroundColor = chipFill(.systemOrange)
            swapChipLabel.stringValue = String(format: "Swap %.1f GB", model.swapGB)
        } else {
            swapChip.isHidden = true
        }
        graphView.needsDisplay = true

        let used = model.appGB + model.wiredGB + model.compressedGB
        statsLabel.stringValue = String(
            format: "Used %.1f · Wired %.1f · Comp %.1f · Free %.1f · Swap %.1f GB",
            used, model.wiredGB, model.compressedGB, model.freeGB, model.swapGB)
        statsLabel.toolTip = String(
            format: "Used %.1f · Wired %.1f · Compressed %.1f · Free %.1f · Swap %.1f GB — LLM %.1f GB",
            used, model.wiredGB, model.compressedGB, model.freeGB, model.swapGB, model.llmGB)

        // section metas — one vocabulary for all backends
        if model.ollamaUp {
            if model.loaded.isEmpty {
                ollamaMetaLabel.stringValue = "idle"
            } else {
                let gb = model.loaded.reduce(0) { $0 + $1.sizeGB }
                ollamaMetaLabel.stringValue = String(format: "%d loaded · %.1f GB", model.loaded.count, gb)
            }
            ollamaMetaLabel.textColor = .secondaryLabelColor
        } else {
            ollamaMetaLabel.stringValue = "offline"
            ollamaMetaLabel.textColor = .systemOrange
        }
        ejectAllButton.isHidden = model.loaded.count < 2
        let mlxRunning = model.mlxServers.filter { $0.running }
        setMeta(mlxMetaLabel, running: mlxRunning.count, known: model.mlxServers.count,
                gb: mlxRunning.reduce(0) { $0 + ($1.weightsGB > 0 ? $1.weightsGB : $1.rssGB) })
        setMeta(splashMetaLabel, running: model.splashServers.filter { $0.running }.count,
                known: model.splashServers.count, gb: model.splashGB)

        // rows — rebuild (with a soft fade) only when structure changes; text-only
        // updates (eject countdown, live memory) rewrite labels in place.
        let recentSpecs = buildRecentSpecs()
        recentSection.isHidden = recentSpecs.isEmpty
        let ollamaSpecs = buildOllamaSpecs()
        let mlxSpecs = buildMlxSpecs()
        let splashSpecs = buildSplashSpecs()
        // Ollama always shows (its placeholder says how to start it); MLX and Splash only when present.
        let showMlx = !model.mlxServers.isEmpty || !model.mlxCached.isEmpty || appDelegate?.hasMlxEnv == true
        let showSplash = !model.splashServers.isEmpty || !model.splashInstalled.isEmpty
            || FileManager.default.isExecutableFile(atPath: AppDelegate.splashCLI) || AppDelegate.splishRoot != nil
        sectionViews["MLX SERVERS"]?.forEach { $0.isHidden = !showMlx }
        sectionViews["SPLASH"]?.forEach { $0.isHidden = !showSplash }
        let allSpecs = recentSpecs + ollamaSpecs + mlxSpecs + splashSpecs
        let sig = allSpecs.map(\.sig).joined(separator: "\u{1}")
        if sig != lastRowSig {
            lastRowSig = sig
            rowCache.removeAll()
            loadPicks.removeAll()
            swapRows(recentRows, specs: recentSpecs)
            swapRows(ollamaRows, specs: ollamaSpecs)
            swapRows(mlxRows, specs: mlxSpecs)
            swapRows(splashRows, specs: splashSpecs)
        } else {
            for spec in allSpecs {
                if case let .model(r) = spec.kind { (rowCache[spec.key] as? ModelRowView)?.update(r) }
            }
        }
        updateHF()
        if tab == .library { updateLibrary() }
        updateTabLabels()
        fitHeight()
    }

    // MARK: tabs

    @objc private func tabChanged() {
        tab = Tab(rawValue: tabControl.selectedSegment) ?? .running
        applyTab()
        if tab == .library, model.library.map({ Date().timeIntervalSince($0.scannedAt) > 20 }) ?? true {
            appDelegate?.rescanLibrary()
        }
        rebuild()
        if tab == .get { window?.makeFirstResponder(hfSearch) }
    }

    private func applyTab() {
        for (i, stack) in tabStacks.enumerated() { stack.isHidden = i != tab.rawValue }
        if tabControl.selectedSegment != tab.rawValue { tabControl.selectedSegment = tab.rawValue }
        graphView.updateAnimation()
        if tab == .library { libSig = "" }
    }

    private func updateTabLabels() {
        var get = "Get"
        if let job = model.hfJob, !job.done, !job.failed {
            get = job.fraction.map { "Get · \(Int($0 * 100))%" } ?? "Get · …"
        }
        if tabControl.label(forSegment: 2) != get { tabControl.setLabel(get, forSegment: 2) }
        let n = model.library?.models.count
        let lib = n.map { "Library · \($0)" } ?? "Library"
        if tabControl.label(forSegment: 1) != lib { tabControl.setLabel(lib, forSegment: 1) }
    }

    static func formatBytes(_ b: Double) -> String {
        let gb = b / 1_073_741_824
        if gb >= 1000 { return String(format: "%.1f TB", gb / 1024) }
        return gb >= 100 ? String(format: "%.0f GB", gb) : String(format: "%.1f GB", gb)
    }

    /// "now", "12m", "3h", "5d", "6w" — compact, for rows.
    static func ago(_ date: Date) -> String {
        let s = max(0, Date().timeIntervalSince(date))
        if s < 60 { return "now" }
        if s < 3600 { return "\(Int(s / 60))m" }
        if s < 86_400 { return "\(Int(s / 3600))h" }
        if s < 14 * 86_400 { return "\(Int(s / 86_400))d" }
        return "\(Int(s / (7 * 86_400)))w"
    }

    private static func loadTitle(_ inst: LibraryInstance) -> String {
        switch inst.backend {
        case .ollama: return "Ollama"
        case .mlx: return "MLX :\(AppDelegate.panelMlxPort)"
        case .splash: return "Splash"
        case .hf: return "Splash (from HF cache)"
        }
    }

    // MARK: recent

    private func buildRecentSpecs() -> [RowSpec] {
        guard let app = appDelegate else { return [] }
        return app.recentItems().map { it in
            let inst = it.instance
            let detail = PanelContentView.loadTitle(inst) + (it.lastUsed.map { " · \(PanelContentView.ago($0))" } ?? "")
            return RowSpec(key: "recent:\(it.key)", kind: .model(ModelRow(
                dot: it.favorite ? .systemYellow : .tertiaryLabelColor, hollow: !it.favorite,
                name: it.name, detail: detail,
                action: RowAction(title: "Load", tooltip: "Load in \(PanelContentView.loadTitle(inst))",
                                  run: { [weak app] in app?.loadInstance(inst) }),
                info: "\(it.name)\n\(inst.place): \(inst.ref)" + (it.favorite ? "\n★ Favorite" : ""))))
        }
    }

    // MARK: library

    private func buildLibrary(in lib: NSStackView) {
        libDiskLabel = uiLabel("Scanning…", 10.5, .regular, .secondaryLabelColor, mono: true)
        libDiskLabel.lineBreakMode = .byTruncatingTail
        addFull(libDiskLabel, to: lib, after: 2)
        libDupLabel = uiLabel("", 10.5, .regular, .systemOrange)
        libDupLabel.lineBreakMode = .byTruncatingTail
        addFull(libDupLabel, to: lib, after: 10)

        libSearch = NSSearchField()
        libSearch.controlSize = .small
        libSearch.font = uiFont(12)
        libSearch.placeholderString = "Filter models"
        libSearch.delegate = self
        libSearch.setContentHuggingPriority(.defaultLow, for: .horizontal)
        libFilter = NSSegmentedControl(labels: ["All", "★", "Unused"], trackingMode: .selectOne,
                                       target: self, action: #selector(libFilterChanged))
        libFilter.controlSize = .small
        libFilter.font = uiFont(10.5, .medium)
        libFilter.selectedSegment = 0
        libFilter.setToolTip("Favorites", forSegment: 1)
        libFilter.setToolTip("Not used for 30 days (favorites excluded)", forSegment: 2)
        libFilter.setContentHuggingPriority(.required, for: .horizontal)
        let filterRow = hstack()
        filterRow.alignment = .centerY
        filterRow.spacing = 8
        filterRow.addArrangedSubview(libSearch)
        filterRow.addArrangedSubview(libFilter)
        addFull(filterRow, to: lib, after: 8)

        libList = NSStackView()
        libList.orientation = .vertical
        libList.alignment = .width
        libList.spacing = 4
        libList.translatesAutoresizingMaskIntoConstraints = false
        let doc = FlippedView()
        doc.translatesAutoresizingMaskIntoConstraints = false
        doc.addSubview(libList)
        libScroll = NSScrollView()
        libScroll.drawsBackground = false
        libScroll.hasVerticalScroller = true
        libScroll.autohidesScrollers = true
        libScroll.scrollerStyle = .overlay
        libScroll.borderType = .noBorder
        libScroll.documentView = doc
        let clip = libScroll.contentView
        libScrollHeight = libScroll.heightAnchor.constraint(equalToConstant: 200)
        NSLayoutConstraint.activate([
            libList.topAnchor.constraint(equalTo: doc.topAnchor),
            libList.leadingAnchor.constraint(equalTo: doc.leadingAnchor),
            libList.trailingAnchor.constraint(equalTo: doc.trailingAnchor),
            libList.bottomAnchor.constraint(equalTo: doc.bottomAnchor),
            doc.topAnchor.constraint(equalTo: clip.topAnchor),
            doc.leadingAnchor.constraint(equalTo: clip.leadingAnchor),
            doc.widthAnchor.constraint(equalTo: clip.widthAnchor),
            libScrollHeight,
        ])
        addFull(libScroll, to: lib, after: 4)
    }

    @objc private func libFilterChanged() {
        libSig = ""
        updateLibrary()
        fitHeight()
    }

    func controlTextDidChange(_ obj: Notification) {
        guard (obj.object as? NSSearchField) === libSearch else { return }
        updateLibrary()
        fitHeight()
    }

    private func updateLibrary() {
        guard let app = appDelegate else { return }
        guard let snap = model.library else {
            libDiskLabel.stringValue = "Scanning models on disk…"
            libDupLabel.isHidden = true
            return
        }
        let fmt = PanelContentView.formatBytes
        libDiskLabel.stringValue = "\(snap.models.count) models · \(fmt(snap.diskBytes)) on disk · \(fmt(snap.freeBytes)) free"
        libDiskLabel.toolTip = "Counted once per file, even when several copies share it.\nScanned \(PanelContentView.ago(snap.scannedAt)) ago"
        let dupes = snap.duplicates
        libDupLabel.isHidden = snap.duplicateBytes < 1_073_741_824
        libDupLabel.stringValue = dupes.count == 1
            ? "\(dupes[0].name) is stored twice — \(fmt(dupes[0].bytes))"
            : "\(fmt(snap.duplicateBytes)) stored twice across \(dupes.count) models (marked 2×)"
        libDupLabel.toolTip = "Same weights, saved once by Ollama and once in the HF cache:\n\n" + dupes.map { d in
            "\(d.name) — \(fmt(d.bytes))\n   Ollama: \(d.ollamaNames.joined(separator: ", "))\n   \(d.advice)"
        }.joined(separator: "\n\n")

        let live = app.liveKeys()
        let query = libSearch.stringValue.trimmingCharacters(in: .whitespaces).lowercased()
        let mode = libFilter.selectedSegment
        let now = Date()
        let stats = { (m: LibraryModel) in app.store.stats(m.key) }
        let shown = snap.models.filter { m in
            if !query.isEmpty, !m.name.lowercased().contains(query), !m.key.contains(query),
               !m.instances.contains(where: { $0.ref.lowercased().contains(query) }) { return false }
            let st = stats(m)
            switch mode {
            case 1: return st?.favorite == true
            case 2:
                guard !live.contains(m.key), st?.favorite != true else { return false }
                return now.timeIntervalSince(st?.lastUsed ?? st?.firstSeen ?? now) > 30 * 86_400
            default: return true
            }
        }.sorted { a, b in
            let la = live.contains(a.key), lb = live.contains(b.key)
            if la != lb { return la }
            let ua = stats(a)?.lastUsed ?? .distantPast, ub = stats(b)?.lastUsed ?? .distantPast
            if ua != ub { return ua > ub }
            return a.sizeBytes > b.sizeBytes
        }
        let favs = mode == 0 ? shown.filter { stats($0)?.favorite == true } : []
        let rest = mode == 0 ? shown.filter { stats($0)?.favorite != true } : shown

        var sig = "\(mode)|\(query)|\(snap.scannedAt.timeIntervalSince1970)|"
        for m in favs + rest {
            let st = stats(m)
            sig += "\(m.key):\(st?.favorite == true):\(live.contains(m.key)):\(st?.lastUsed.map { PanelContentView.ago($0) } ?? "-");"
        }
        if let p = pendingDelete, let m = snap.models.first(where: { $0.key == p.key }) {
            let blockers = m.instances.filter { p.ids.contains($0.id) }.compactMap { app.deleteBlocker($0) }
            sig += "del:\(p.key):\(p.ids):\(p.busy):\(p.error ?? ""):\(blockers)"
        }
        guard sig != libSig else { return }
        libSig = sig

        libList.arrangedSubviews.forEach { $0.removeFromSuperview() }
        func add(_ v: NSView) {
            libList.addArrangedSubview(v)
            v.widthAnchor.constraint(equalTo: libList.widthAnchor).isActive = true
        }
        func header(_ text: String) {
            let h = hstack()
            h.edgeInsets = NSEdgeInsets(top: libList.arrangedSubviews.isEmpty ? 2 : 8, left: 2, bottom: 2, right: 0)
            h.addArrangedSubview(eyebrow(text))
            add(h)
        }
        func row(_ m: LibraryModel) {
            if let p = pendingDelete, p.key == m.key { add(makeConfirmRow(m, p, snap: snap)) }
            else { add(makeLibraryRow(m, live: live.contains(m.key))) }
        }
        if shown.isEmpty {
            let text = !query.isEmpty ? "No models match “\(query)”"
                : mode == 1 ? "No favorites yet — star a model to pin it"
                : mode == 2 ? "Nothing unused for 30 days" : "No models on disk"
            add(ModelRowView(ModelRow(dot: .tertiaryLabelColor, hollow: true, name: text, dimName: true)))
        }
        if !favs.isEmpty {
            header("FAVORITES")
            favs.forEach(row)
            if !rest.isEmpty { header("ALL MODELS") }
        }
        rest.forEach(row)

        libList.layoutSubtreeIfNeeded()
        libScrollHeight.constant = min(ceil(libList.fittingSize.height), 430)
    }

    private func badge(_ b: Backend) -> NSView {
        let box = NSView()
        box.wantsLayer = true
        box.layer?.cornerRadius = 3.5
        box.layer?.backgroundColor = NSColor.textColor.withAlphaComponent(0.08).cgColor
        box.toolTip = b.title
        box.translatesAutoresizingMaskIntoConstraints = false
        let l = uiLabel(b.badge, 9, .bold, .secondaryLabelColor, mono: true)
        l.translatesAutoresizingMaskIntoConstraints = false
        box.addSubview(l)
        NSLayoutConstraint.activate([
            box.widthAnchor.constraint(equalToConstant: 15),
            box.heightAnchor.constraint(equalToConstant: 15),
            l.centerXAnchor.constraint(equalTo: box.centerXAnchor),
            l.centerYAnchor.constraint(equalTo: box.centerYAnchor),
        ])
        return box
    }

    private func makeLibraryRow(_ m: LibraryModel, live: Bool) -> NSView {
        guard let app = appDelegate else { return NSView() }
        let st = app.store.stats(m.key)
        let fav = st?.favorite == true
        let row = HoverRowView()
        let h = hstack()
        h.alignment = .centerY
        h.spacing = 6
        h.edgeInsets = NSEdgeInsets(top: 4, left: 6, bottom: 4, right: 4)
        h.translatesAutoresizingMaskIntoConstraints = false
        row.addSubview(h)
        NSLayoutConstraint.activate([
            h.topAnchor.constraint(equalTo: row.topAnchor),
            h.bottomAnchor.constraint(equalTo: row.bottomAnchor),
            h.leadingAnchor.constraint(equalTo: row.leadingAnchor),
            h.trailingAnchor.constraint(equalTo: row.trailingAnchor),
        ])

        let star = ClosureButton(title: "")
        star.isBordered = false
        star.bezelStyle = .inline
        star.imagePosition = .imageOnly
        star.image = sysSymbol(fav ? "star.fill" : "star", fallback: "star", size: 12, weight: .medium)
        star.contentTintColor = fav ? .systemYellow : .tertiaryLabelColor
        star.toolTip = fav ? "Remove from favorites" : "Add to favorites"
        star.onAction = { [weak app] in app?.toggleFavorite(key: m.key, name: m.name) }
        star.translatesAutoresizingMaskIntoConstraints = false
        star.widthAnchor.constraint(equalToConstant: 18).isActive = true
        h.addView(star, in: .leading)

        let middle = NSMutableParagraphStyle()
        middle.lineBreakMode = .byTruncatingMiddle
        let name = NSTextField(labelWithAttributedString: NSAttributedString(string: m.name, attributes: [
            .font: uiFont(12.5), .foregroundColor: NSColor.labelColor, .paragraphStyle: middle,
        ]))
        name.maximumNumberOfLines = 1
        name.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        var tip = [m.name] + m.instances.map { "\($0.place): \($0.ref)" }
        if let st, st.loadCount > 0 || st.secondsLoaded > 0 {
            tip.append(String(format: "Loaded %d× · %.1f h in memory · peak %.1f GB", st.loadCount, st.secondsLoaded / 3600, st.peakGB))
        }
        if let used = st?.lastUsed { tip.append("Last used \(PanelContentView.ago(used)) ago") }
        else { tip.append("Not used since tracking started") }
        name.toolTip = tip.joined(separator: "\n")
        h.addView(name, in: .leading)

        if live {
            let dot = NSView()
            dot.wantsLayer = true
            dot.layer?.cornerRadius = 3
            dot.layer?.backgroundColor = NSColor.systemGreen.cgColor
            dot.toolTip = "In memory now"
            dot.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([dot.widthAnchor.constraint(equalToConstant: 6), dot.heightAnchor.constraint(equalToConstant: 6)])
            h.addView(dot, in: .leading)
        }
        var seen = Set<Backend>()
        for i in m.instances where seen.insert(i.backend).inserted { h.addView(badge(i.backend), in: .leading) }
        if let d = model.library?.duplicates.first(where: { $0.key == m.key }) {
            let twice = uiLabel("2×", 9.5, .bold, .systemOrange, mono: true)
            twice.toolTip = "Stored twice — \(PanelContentView.formatBytes(d.bytes)) extra\n\(d.advice)"
            h.addView(twice, in: .leading)
        }

        let used = st?.lastUsed.map { PanelContentView.ago($0) } ?? "—"
        let detail = uiLabel("\(PanelContentView.formatBytes(m.sizeBytes)) · \(used)", 10.5, .regular, .secondaryLabelColor, mono: true)
        detail.setContentHuggingPriority(.required, for: .horizontal)
        detail.setContentCompressionResistancePriority(.required, for: .horizontal)
        h.addView(detail, in: .trailing)

        let more = NSPopUpButton(frame: .zero, pullsDown: true)
        more.isBordered = false
        more.controlSize = .small
        (more.cell as? NSPopUpButtonCell)?.arrowPosition = .noArrow
        more.menu = libraryMenu(m, live: live)
        more.toolTip = "Load, reveal, copy, delete"
        more.translatesAutoresizingMaskIntoConstraints = false
        more.widthAnchor.constraint(equalToConstant: 26).isActive = true
        h.addView(more, in: .trailing)
        return row
    }

    private func menuItem(_ title: String, enabled: Bool = true, _ run: @escaping () -> Void) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: enabled ? #selector(runMenuAction(_:)) : nil, keyEquivalent: "")
        item.target = self
        item.representedObject = MenuAction(run)
        item.isEnabled = enabled
        return item
    }

    @objc private func runMenuAction(_ item: NSMenuItem) {
        (item.representedObject as? MenuAction)?.run()
    }

    private func libraryMenu(_ m: LibraryModel, live: Bool) -> NSMenu {
        let menu = NSMenu()
        menu.autoenablesItems = false
        let title = NSMenuItem(title: "", action: nil, keyEquivalent: "")
        title.image = sysSymbol("ellipsis.circle", fallback: "ellipsis", size: 12)
        menu.addItem(title)
        guard let app = appDelegate else { return menu }
        let ollamaCount = m.instances.filter { $0.backend == .ollama }.count
        func label(_ i: LibraryInstance) -> String {
            PanelContentView.loadTitle(i) + (i.backend == .ollama && ollamaCount > 1 ? " — \(i.ref)" : "")
        }

        for i in m.instances {
            menu.addItem(menuItem("Load in \(label(i))", enabled: !live) { [weak app] in app?.loadInstance(i) })
        }
        menu.addItem(.separator())
        let fav = app.store.isFavorite(m.key)
        menu.addItem(menuItem(fav ? "Remove from Favorites" : "Add to Favorites") { [weak app] in
            app?.toggleFavorite(key: m.key, name: m.name)
        })
        if let i = app.preferredInstance(m) {
            let path: String? = {
                switch i.backend {
                case .ollama: return i.files.max { $0.value < $1.value }?.key
                case .mlx: return i.ref
                case .splash: return i.links.first
                case .hf: return i.repoDir
                }
            }()
            if let path {
                menu.addItem(menuItem("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                })
            }
            let id = i.backend == .mlx ? (i.repo ?? i.ref) : i.ref
            menu.addItem(menuItem("Copy Model ID") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(id, forType: .string)
            })
        }
        if let repo = m.repo, let url = URL(string: "https://huggingface.co/\(repo)") {
            menu.addItem(menuItem("Open on Hugging Face") { NSWorkspace.shared.open(url) })
        }
        menu.addItem(.separator())
        for i in m.instances {
            let title = "Delete from \(i.place)" + (i.backend == .ollama && ollamaCount > 1 ? " — \(i.ref)" : "") + "…"
            menu.addItem(menuItem(title) { [weak self] in self?.askDelete(m, [i]) })
        }
        if m.instances.count > 1 {
            menu.addItem(menuItem("Delete Everywhere…") { [weak self] in self?.askDelete(m, m.instances) })
        }
        return menu
    }

    private func askDelete(_ m: LibraryModel, _ targets: [LibraryInstance]) {
        pendingDelete = PendingDelete(key: m.key, ids: targets.map(\.id))
        libSig = ""
        updateLibrary()
        fitHeight()
    }

    /// Inline confirmation: what goes, how much space comes back, and anything that blocks it.
    private func makeConfirmRow(_ m: LibraryModel, _ p: PendingDelete, snap: LibrarySnapshot) -> NSView {
        guard let app = appDelegate else { return NSView() }
        let targets = m.instances.filter { p.ids.contains($0.id) }
        let freed = LibraryScanner.bytesFreed(targets, in: snap)
        let blocker = targets.compactMap { app.deleteBlocker($0) }.first
        let everywhere = targets.count == m.instances.count && m.instances.count > 1

        let row = HoverRowView(height: 56)
        row.layer?.backgroundColor = NSColor.systemRed.withAlphaComponent(0.09).cgColor
        row.layer?.borderWidth = 0.5
        row.layer?.borderColor = NSColor.systemRed.withAlphaComponent(0.35).cgColor

        let what = everywhere ? "Delete everywhere"
            : "Delete from \(targets.first?.place ?? "?")"
        let title = uiLabel("\(what) — frees \(PanelContentView.formatBytes(freed))", 12, .semibold)
        title.lineBreakMode = .byTruncatingTail
        title.toolTip = m.name + "\n" + targets.map { "\($0.place): \($0.ref)" }.joined(separator: "\n")

        var note: String
        var noteColor = NSColor.secondaryLabelColor
        if let err = p.error { note = err; noteColor = .systemOrange }
        else if p.busy { note = "Deleting…" }
        else if let blocker { note = blocker; noteColor = .systemOrange }
        else {
            var parts: [String] = []
            if app.store.isFavorite(m.key) { parts.append("★ Favorite") }
            if freed < max(m.sizeBytes * 0.05, 1) { parts.append("Shares its files with the other copies") }
            else if everywhere { parts.append("Removes every copy listed") }
            else {
                switch targets.first?.backend {
                case .ollama: parts.append("Removes it from Ollama")
                case .splash: parts.append("Removes the Splash install and its HF cache files")
                case .hf: parts.append("Removes the GGUF from the HF cache")
                case .mlx: parts.append("Removes the model folder from the HF cache")
                case nil: break
                }
            }
            note = parts.joined(separator: " · ")
        }
        let noteLabel = uiLabel(note, 10.5, .regular, noteColor)
        noteLabel.lineBreakMode = .byTruncatingTail
        noteLabel.toolTip = note

        let text = NSStackView(views: [title, noteLabel])
        text.orientation = .vertical
        text.alignment = .leading
        text.spacing = 2
        for v in [title, noteLabel] { v.setContentCompressionResistancePriority(.defaultLow, for: .horizontal) }

        let cancel = rowButton(p.error == nil ? "Cancel" : "Close", width: 54)
        cancel.isEnabled = !p.busy
        cancel.onAction = { [weak self] in
            self?.pendingDelete = nil
            self?.libSig = ""
            self?.updateLibrary()
            self?.fitHeight()
        }
        let delete = rowButton("Delete", width: 54)
        delete.contentTintColor = .systemRed
        delete.hasDestructiveAction = true
        delete.isEnabled = blocker == nil && !p.busy && p.error == nil
        delete.onAction = { [weak self, weak app] in
            guard let self, let app else { return }
            self.pendingDelete?.busy = true
            self.libSig = ""
            self.updateLibrary()
            app.deleteFromLibrary(targets, key: m.key, name: m.name) { [weak self] error in
                guard let self else { return }
                if let error {
                    self.pendingDelete?.busy = false
                    self.pendingDelete?.error = error
                } else {
                    self.pendingDelete = nil
                }
                self.libSig = ""
                self.updateLibrary()
                self.fitHeight()
            }
        }

        let h = hstack()
        h.alignment = .centerY
        h.spacing = 6
        h.edgeInsets = NSEdgeInsets(top: 6, left: 10, bottom: 6, right: 8)
        h.translatesAutoresizingMaskIntoConstraints = false
        h.addView(text, in: .leading)
        h.addView(cancel, in: .trailing)
        h.addView(delete, in: .trailing)
        row.addSubview(h)
        NSLayoutConstraint.activate([
            h.topAnchor.constraint(equalTo: row.topAnchor),
            h.bottomAnchor.constraint(equalTo: row.bottomAnchor),
            h.leadingAnchor.constraint(equalTo: row.leadingAnchor),
            h.trailingAnchor.constraint(equalTo: row.trailingAnchor),
        ])
        return row
    }

    private func setMeta(_ label: NSTextField, running: Int, known: Int, gb: Double) {
        if running > 0 {
            label.stringValue = String(format: "%d running · %.1f GB", running, gb)
        } else {
            label.stringValue = known > 0 ? "stopped" : "none"
        }
        label.textColor = .secondaryLabelColor
    }

    private func fitHeight() {
        layoutSubtreeIfNeeded()
        let h = ceil(rootStack.fittingSize.height) + 28
        guard abs(h - fittedHeight) > 0.5 else { return }
        fittedHeight = h
        appDelegate?.contentHeightChanged(h, for: self)
    }

    // MARK: row specs

    private func placeholder(_ key: String, _ title: String, hint: String) -> RowSpec {
        RowSpec(key: key, kind: .model(ModelRow(dot: .tertiaryLabelColor, hollow: true, name: title,
                                                info: hint, dimName: true)))
    }

    private static func ejectCountdown(_ secs: Int) -> String {
        secs >= 120 ? "Ejects in \(secs / 60) min" : String(format: "Ejects in %d:%02d", secs / 60, secs % 60)
    }

    private func buildOllamaSpecs() -> [RowSpec] {
        guard model.ollamaUp else {
            return [placeholder("ollama-off", "Ollama isn't running", hint: "Start Ollama to load models.")]
        }
        var specs: [RowSpec] = []
        if model.loaded.isEmpty {
            specs.append(placeholder("ollama-empty", "No models loaded",
                                     hint: model.installed.isEmpty
                                         ? "Pull a model below to load it here."
                                         : "Load a model — it stays in memory for 30 minutes."))
        }
        for m in model.loaded {
            var detail = ""
            var ejecting = false
            if let exp = m.expiresAt {
                let secs = Int(exp.timeIntervalSinceNow)
                if secs > 0 { detail = PanelContentView.ejectCountdown(secs) } else { ejecting = true; detail = "Ejecting…" }
            }
            var info = m.name
            if let ctx = m.contextLength { info += "\ncontext \(AppDelegate.formatTokens(ctx))" }
            specs.append(RowSpec(key: "ollama:\(m.name)", kind: .model(ModelRow(
                dot: .systemGreen, pulse: ejecting, name: shortModelName(m.name),
                size: String(format: "%.1f GB", m.sizeGB), detail: detail,
                action: RowAction(title: "Eject", enabled: !ejecting,
                                  tooltip: ejecting ? "Unloading…" : "Unload from memory",
                                  run: { [weak self] in self?.appDelegate?.unloadModel(m.name) }),
                info: info))))
        }
        let loadedNames = Set(model.loaded.map { $0.name })
        let loadable = model.installed.filter { !loadedNames.contains($0) }.sorted()
        if !loadable.isEmpty, let app = appDelegate {
            let star = { app.isFavorite(.ollama, $0) }
            specs.append(RowSpec(key: "ollama-load|\(loadable.map { (star($0) ? "*" : "") + $0 }.joined(separator: "|"))", kind: .load(
                names: loadable, tooltip: "Load into memory for 30 minutes", star: star,
                pick: { [weak self] in self?.appDelegate?.loadModel($0) })))
        }
        return specs
    }

    private func buildMlxSpecs() -> [RowSpec] {
        var specs = buildMlxServerSpecs()
        let served = Set(model.mlxServers.filter(\.running).compactMap(\.modelPath))
        let cached = model.mlxCached.filter { !served.contains($0.path) }
        if !cached.isEmpty, let app = appDelegate {
            let byRepo = Dictionary(cached.map { ($0.repo, $0) }, uniquingKeysWith: { a, _ in a })
            let star = { (repo: String) in byRepo[repo].map { app.isFavorite(.mlx, $0.path) } ?? false }
            specs.append(RowSpec(key: "mlx-load|\(cached.map { (star($0.repo) ? "*" : "") + $0.repo }.joined(separator: "|"))", kind: .load(
                names: cached.map(\.repo),
                tooltip: "Serve a downloaded MLX model on :\(AppDelegate.panelMlxPort) — replaces the model running there",
                star: star,
                pick: { [weak self] repo in
                    guard let m = byRepo[repo] else { return }
                    self?.appDelegate?.serveInMlx(path: m.path, repo: m.repo, server: m.server)
                })))
        }
        return specs
    }

    private func buildMlxServerSpecs() -> [RowSpec] {
        if model.mlxServers.isEmpty {
            return [placeholder("mlx-empty", "No MLX servers",
                                hint: "Pick a model below, or configure servers in mlx_servers.json (see README).")]
        }
        return model.mlxServers.compactMap { srv -> RowSpec? in
            var info = "\(srv.name)\nport \(srv.port)" + (srv.pid.map { "\npid \($0)" } ?? "")
            if let path = srv.modelPath { info += "\n\(path)" }
            if srv.panel { info += "\nPanel slot — picking another model replaces it" }
            if srv.running {
                let gb = srv.weightsGB > 0 ? srv.weightsGB : srv.rssGB
                return RowSpec(key: "mlx:\(srv.name):\(srv.port)", kind: .model(ModelRow(
                    dot: .systemPurple, name: shortModelName(srv.name), port: srv.port,
                    size: gb > 0 ? String(format: "%.1f GB", gb) : "",
                    action: RowAction(title: "Stop", tooltip: "Stop server",
                                      run: { [weak self] in self?.appDelegate?.stopMlx(srv) }),
                    info: info)))
            } else if srv.configured {
                return RowSpec(key: "mlx:\(srv.name):\(srv.port)", kind: .model(ModelRow(
                    dot: .systemGray, name: shortModelName(srv.name), port: srv.port, detail: "Stopped",
                    action: RowAction(title: "Start", tooltip: "Start server",
                                      run: { [weak self] in self?.appDelegate?.startMlx(srv) }),
                    info: info)))
            }
            return nil // unconfigured, not running: invisible, as before
        }
    }

    private func buildSplashSpecs() -> [RowSpec] {
        var specs: [RowSpec] = []
        if model.splashServers.isEmpty {
            specs.append(placeholder("splash-empty", "No Splash servers",
                                     hint: "Run “splash serve”, add a LaunchAgent, or pick a model below."))
        }
        for srv in model.splashServers {
            var info = [srv.displayName, "port \(srv.port)"]
            if let pid = srv.pid { info.append("pid \(pid)") }
            if let ctx = srv.contextTokens { info.append("context \(AppDelegate.formatTokens(ctx))") }
            if srv.peakGB > 0 { info.append(String(format: "peak %.1f GB", srv.peakGB)) }
            info.append(srv.label.map { "LaunchAgent \($0)" } ?? "started by hand (\(srv.engine.lowercased()) serve)")
            if srv.engine == "Splish" { info.append("engine: Splish (M5-tuned Splash fork)") }
            if let agent = srv.agentModel, srv.running, agent != srv.displayName {
                info.append("agent model: \(agent)")
            }
            if let log = srv.lastLog { info.append(log) }

            guard srv.running else {
                let blocker = model.splashServers.first { $0.running && $0.port == srv.port }
                specs.append(RowSpec(key: "splash:\(srv.id):stopped", kind: .model(ModelRow(
                    dot: .systemGray, name: srv.shortName, port: srv.port, detail: "Stopped",
                    action: RowAction(
                        title: blocker == nil ? "Start" : "Switch",
                        tooltip: blocker.map { "Stop \($0.shortName) and start this on :\(srv.port)" }
                            ?? "Start via launchd",
                        run: { [weak self] in self?.appDelegate?.startSplash(srv) }),
                    info: info.joined(separator: "\n")))))
                continue
            }
            let detail = srv.ready ? (srv.busy ? "Busy" : (srv.engine == "Splish" ? "Splish" : ""))
                                   : (srv.lastLog.map { _ in "Starting…" } ?? "Starting…")
            specs.append(RowSpec(key: "splash:\(srv.id):\(srv.ready ? "up" : "starting")", kind: .model(ModelRow(
                dot: srv.ready ? .systemTeal : .systemOrange, pulse: !srv.ready,
                name: srv.shortName, port: srv.port,
                size: srv.memGB > 0 ? String(format: "%.1f GB", srv.memGB) : "", detail: detail,
                action: RowAction(title: "Stop", tooltip: "Stop server",
                                  run: { [weak self] in self?.appDelegate?.stopSplash(srv) }),
                info: info.joined(separator: "\n")))))
        }
        let serving = Set(model.splashServers.filter(\.running).compactMap(\.model))
        let loadable = model.splashInstalled.filter { !serving.contains($0) }
        if !loadable.isEmpty, let app = appDelegate {
            let star = { app.isFavorite(.splash, $0) }
            specs.append(RowSpec(key: "splash-load|\(loadable.map { (star($0) ? "*" : "") + $0 }.joined(separator: "|"))", kind: .load(
                names: loadable,
                tooltip: "Serve on :\(AppDelegate.panelSplashPort) — replaces the model running there"
                    + (AppDelegate.splishRoot != nil ? " (with \(app.preferredSplashEngine) unless installed in only one)" : ""),
                star: star,
                pick: { [weak self] in self?.appDelegate?.loadSplashModel($0) })))
        }
        return specs
    }

    // MARK: rows

    private func swapRows(_ stack: NSStackView, specs: [RowSpec]) {
        stack.arrangedSubviews.forEach { $0.removeFromSuperview() }
        var prev: NSView?
        for spec in specs {
            let view: NSView
            switch spec.kind {
            case let .model(r):
                view = ModelRowView(r)
            case let .load(names, tooltip, star, pick):
                view = makeLoadRow(names, tooltip: tooltip, star: star, pick: pick)
            }
            view.wantsLayer = true
            if !reduceMotion, let layer = view.layer {
                let fade = CABasicAnimation(keyPath: "opacity")
                fade.fromValue = 0.25
                fade.toValue = 1
                fade.duration = 0.2
                fade.timingFunction = CAMediaTimingFunction(name: .easeOut)
                layer.add(fade, forKey: "fade-in")
            }
            stack.addArrangedSubview(view)
            view.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
            if let prev { stack.setCustomSpacing(4, after: prev) }
            prev = view
            rowCache[spec.key] = view
        }
    }

    private var loadPicks: [ObjectIdentifier: (String) -> Void] = [:]

    /// A pull-down whose fixed title is the action, so it never reads as a current selection.
    private func makeLoadRow(_ names: [String], tooltip: String, star: (String) -> Bool,
                             pick: @escaping (String) -> Void) -> NSView {
        let wrap = hstack()
        wrap.alignment = .centerY
        wrap.edgeInsets = NSEdgeInsets(top: 2, left: 10, bottom: 2, right: 10)
        let popup = NSPopUpButton(frame: .zero, pullsDown: true)
        popup.controlSize = .small
        popup.font = uiFont(11, .medium)
        popup.addItem(withTitle: "Load Model…")
        // favorites first, starred
        for name in names.filter(star) + names.filter({ !star($0) }) {
            popup.addItem(withTitle: (star(name) ? "★ " : "") + shortModelName(name))
            popup.lastItem?.representedObject = name
            popup.lastItem?.toolTip = name
        }
        popup.target = self
        popup.action = #selector(loadPicked(_:))
        popup.toolTip = tooltip
        loadPicks[ObjectIdentifier(popup)] = pick
        wrap.addView(popup, in: .trailing)
        return wrap
    }

    @objc private func loadPicked(_ sender: NSPopUpButton) {
        guard let name = sender.selectedItem?.representedObject as? String else { return }
        loadPicks[ObjectIdentifier(sender)]?(name)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(CommandLine.arguments.contains("--window") ? .regular : .accessory)
app.run()