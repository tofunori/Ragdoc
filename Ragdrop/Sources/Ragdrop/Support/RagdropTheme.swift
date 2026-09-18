import AppKit
import SwiftUI

enum RagdropAppearance: String, CaseIterable, Identifiable {
    case light, dark, system
    var id: Self { self }
    var title: String {
        switch self { case .light: "Light"; case .dark: "Dark"; case .system: "System" }
    }
    var colorScheme: ColorScheme? {
        switch self { case .light: .light; case .dark: .dark; case .system: nil }
    }
    static var defaults: UserDefaults {
        #if RAGDROP_THEME_DEMO || RAGDROP_REVIEW_DEMO
        UserDefaults(suiteName: "com.tofunori.ragdrop.theme-demo")!
        #else
        .standard
        #endif
    }
}

/// The same role colors are shared by SwiftUI, AppKit and generated HTML.
struct RagdropPalette: Sendable {
    let canvas, panel, raised, sidebar, text, secondary, accent, onAccent, link, line, signal: String
    static let light = RagdropPalette(canvas: "F7F8F8", panel: "FFFFFF", raised: "E9ECEE", sidebar: "F0F2F2",
        text: "24292C", secondary: "626C73", accent: "2E363B", onAccent: "FFFFFF", link: "426A6A", line: "D9DEDF", signal: "527A79")
    static let dark = RagdropPalette(canvas: "15191C", panel: "1C2125", raised: "293137", sidebar: "191E22",
        text: "E5EAED", secondary: "AAB3BA", accent: "D6DEE2", onAccent: "15191C", link: "9DC6C6", line: "394249", signal: "89B6B6")
}

enum RagdropTheme {
    static func nsColor(light: String, dark: String) -> NSColor {
        NSColor(name: nil) { appearance in
            NSColor(hex: appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light)
        }
    }
    private static func color(_ role: KeyPath<RagdropPalette, String>) -> Color {
        Color(nsColor: nsColor(light: RagdropPalette.light[keyPath: role], dark: RagdropPalette.dark[keyPath: role]))
    }
    static let canvas = color(\.canvas)
    static let panel = color(\.panel)
    static let raised = color(\.raised)
    static let sidebar = color(\.sidebar)
    static let text = color(\.text)
    static let secondary = color(\.secondary)
    static let accent = color(\.accent)
    static let onAccent = color(\.onAccent)
    static let link = color(\.link)
    static let line = color(\.line)
    static let signal = color(\.signal)
    static let success = Color(nsColor: nsColor(light: "21734B", dark: "8CD4AC"))
    static let warning = Color(nsColor: nsColor(light: "8B5B00", dark: "D8B878"))
    static let error = Color(nsColor: nsColor(light: "B73334", dark: "F19997"))
    static let radius: CGFloat = 5
    static let pagePadding: CGFloat = 28
    static let title = Font.system(size: 26, weight: .semibold)
    static let nsCanvas = nsColor(light: RagdropPalette.light.canvas, dark: RagdropPalette.dark.canvas)
    static let nsText = nsColor(light: RagdropPalette.light.text, dark: RagdropPalette.dark.text)
    static func cssVariables(_ palette: RagdropPalette) -> String {
        "--canvas:#\(palette.canvas);--panel:#\(palette.panel);--raised:#\(palette.raised);--text:#\(palette.text);--secondary:#\(palette.secondary);--accent:#\(palette.accent);--link:#\(palette.link);--line:#\(palette.line);"
    }
}

private extension NSColor {
    convenience init(hex: String) {
        let value = UInt32(hex, radix: 16) ?? 0
        self.init(srgbRed: CGFloat((value >> 16) & 255) / 255,
                  green: CGFloat((value >> 8) & 255) / 255,
                  blue: CGFloat(value & 255) / 255, alpha: 1)
    }
}

struct RagdropSurface: ViewModifier {
    @AppStorage("appearanceMode", store: RagdropAppearance.defaults) private var mode = RagdropAppearance.light.rawValue
    func body(content: Content) -> some View {
        content
            .foregroundStyle(RagdropTheme.text)
            .tint(RagdropTheme.accent)
            .background(RagdropTheme.canvas)
            .environment(\.locale, Locale(identifier: "en"))
            .preferredColorScheme((RagdropAppearance(rawValue: mode) ?? .light).colorScheme)
    }
}

extension View {
    func ragdropSurface() -> some View { modifier(RagdropSurface()) }
    func ragdropPanel() -> some View {
        background(RagdropTheme.panel, in: .rect(cornerRadius: RagdropTheme.radius))
            .overlay { RoundedRectangle(cornerRadius: RagdropTheme.radius).strokeBorder(RagdropTheme.line) }
    }
}

struct PageHeading: View {
    let title: String
    let subtitle: String
    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(RagdropTheme.title).tracking(-0.4)
            if !subtitle.isEmpty { Text(subtitle).font(.body).foregroundStyle(RagdropTheme.secondary) }
        }
    }
}

struct InlineNotice: View {
    let text: String
    var symbol = "info.circle"
    var isError = false
    var body: some View {
        Label(text, systemImage: symbol)
            .font(.callout)
            .foregroundStyle(isError ? RagdropTheme.warning : RagdropTheme.secondary)
            .padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .ragdropPanel()
    }
}

/// Remains legible when the window is inactive; native buttons retain keyboard semantics.
struct RagdropPrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    @Environment(\.isFocused) private var focused
    @Environment(\.controlSize) private var size
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: size == .large ? 14 : 13, weight: .semibold))
            .padding(.horizontal, size == .large ? 20 : 14)
            .padding(.vertical, size == .large ? 11 : 8)
            .foregroundStyle(enabled ? RagdropTheme.onAccent : RagdropTheme.secondary)
            .background(enabled ? (configuration.isPressed ? RagdropTheme.accent.opacity(0.85) : RagdropTheme.accent) : RagdropTheme.raised,
                        in: .rect(cornerRadius: 5))
            .overlay { RoundedRectangle(cornerRadius: 5).strokeBorder(focused ? RagdropTheme.link : RagdropTheme.line, lineWidth: focused ? 2 : 1) }
    }
}

struct RagdropSecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    @Environment(\.isFocused) private var focused
    @Environment(\.controlSize) private var size
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: size == .large ? 14 : 13, weight: .semibold))
            .padding(.horizontal, size == .large ? 20 : 14)
            .padding(.vertical, size == .large ? 11 : 8)
            .foregroundStyle(enabled ? RagdropTheme.text : RagdropTheme.secondary)
            .background(configuration.isPressed ? RagdropTheme.raised : RagdropTheme.panel, in: .rect(cornerRadius: 5))
            .overlay { RoundedRectangle(cornerRadius: 5).strokeBorder(focused ? RagdropTheme.link : RagdropTheme.line, lineWidth: focused ? 2 : 1) }
    }
}

/// Load the exact bundled application icon; never substitute a decorative logo.
struct RagdropBrandIcon: View {
    private static let image: NSImage? = Bundle.main.url(forResource: "RagdropIcon", withExtension: "icns")
        .flatMap { NSImage(contentsOf: $0) }
    var body: some View {
        if let image = Self.image {
            Image(nsImage: image).resizable().interpolation(.high).scaledToFit().accessibilityHidden(true)
        }
    }
}
