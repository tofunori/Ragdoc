import AppKit
import SwiftUI
import WebKit

struct HTMLPreviewWebView: NSViewRepresentable {
    @Environment(\.colorScheme) private var colorScheme
    let html: String
    let baseURL: URL?

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = false
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.setValue(false, forKey: "drawsBackground")
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        webView.appearance = NSAppearance(named: colorScheme == .dark ? .darkAqua : .aqua)
        let themed = html.replacingOccurrences(of: "<html>", with: "<html data-theme=\"\(colorScheme == .dark ? "dark" : "light")\">")
        guard context.coordinator.lastHTML != themed else { return }
        context.coordinator.lastHTML = themed
        webView.loadHTMLString(themed, baseURL: baseURL)
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var lastHTML = ""

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void
        ) {
            guard let url = navigationAction.request.url else {
                decisionHandler(.cancel)
                return
            }
            let isLocalDocument = url.isFileURL || url.scheme == "about"
            decisionHandler(isLocalDocument ? .allow : .cancel)
        }
    }
}

struct SourceTextView: NSViewRepresentable {
    @Environment(\.colorScheme) private var colorScheme
    let text: String

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.drawsBackground = false

        let textView = NSTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = true
        textView.backgroundColor = RagdropTheme.nsCanvas
        textView.textColor = RagdropTheme.nsText
        textView.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        textView.textContainerInset = NSSize(width: 22, height: 22)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        let appearance = NSAppearance(named: colorScheme == .dark ? .darkAqua : .aqua)
        scrollView.appearance = appearance
        guard let textView = scrollView.documentView as? NSTextView else { return }
        textView.appearance = appearance
        textView.backgroundColor = RagdropTheme.nsCanvas
        textView.textColor = RagdropTheme.nsText
        if textView.string != text { textView.string = text }
    }
}
