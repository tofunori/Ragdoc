import ImageIO
import SwiftUI

struct ReviewArtifactView: View {
    let artifact: ReviewArtifact
    let bundleURL: URL?
    let canNavigate: Bool
    let onPage: () -> Void
    @State private var thumbnail: CGImage?
    @State private var html = ""
    @State private var imageFailed = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(artifact.label).font(.headline)
                Spacer()
                if let page = artifact.page {
                    Button("Page \(page)", action: onPage).disabled(!canNavigate)
                        .help(canNavigate ? "Afficher cette page du PDF" : "Page déclarée par le convertisseur, lien non vérifié")
                }
            }
            if let thumbnail {
                Image(decorative: thumbnail, scale: 1).resizable().scaledToFit().frame(maxHeight: 650)
            } else if imageFailed {
                Label("Image indisponible", systemImage: "photo.badge.exclamationmark").foregroundStyle(RagdropTheme.secondary)
            }
            if !html.isEmpty {
                HTMLPreviewWebView(html: html, baseURL: bundleURL).frame(height: 280)
            }
            if let caption = artifact.caption, !caption.isEmpty { Text(caption).font(.callout).foregroundStyle(RagdropTheme.secondary) }
        }
        .padding(14)
        .ragdropPanel()
        .task(id: artifact.id) {
            if let path = artifact.image, let bundleURL {
                let url = bundleURL.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
                let root = bundleURL.standardizedFileURL.resolvingSymlinksInPath().path + "/"
                if url.path.hasPrefix(root) {
                    let image = await Task.detached(priority: .utility) {
                        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return Optional<CGImage>.none }
                        return CGImageSourceCreateThumbnailAtIndex(source, 0, [
                            kCGImageSourceCreateThumbnailFromImageAlways: true,
                            kCGImageSourceThumbnailMaxPixelSize: 1600,
                            kCGImageSourceCreateThumbnailWithTransform: true
                        ] as CFDictionary)
                    }.value
                    guard !Task.isCancelled else { return }
                    thumbnail = image
                }
                imageFailed = thumbnail == nil
            }
            if let body = artifact.body, !body.isEmpty, (artifact.image == nil || imageFailed) {
                let rendered = try? await MarkdownHTMLRenderer.render(text: String(body.prefix(100_000)))
                guard !Task.isCancelled else { return }
                html = rendered?.html ?? ""
            }
        }
    }
}
