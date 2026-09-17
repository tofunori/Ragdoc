import AppKit
import SwiftUI

struct MarkdownPreviewView: View {
    let job: ImportJob
    let onApprove: () -> Void
    let onReject: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var markdown = ""
    @State private var html = ""
    @State private var loadError: String?
    @State private var isLoading = true
    @State private var mode = PreviewMode.rendered
    @State private var visualArtifacts: [PreviewArtifact] = []

    private enum PreviewMode: String, CaseIterable, Identifiable {
        case rendered = "Rendu"
        case source = "Source"
        case visuals = "Tableaux et figures"
        var id: Self { self }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(job.displayName).font(.headline)
                    Text("Résultat produit par le convertisseur PDF")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Picker("Affichage", selection: $mode) {
                    ForEach(PreviewMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 340)
            }
            .padding()

            Divider()

            Group {
                if isLoading {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("Préparation de l’aperçu…")
                            .foregroundStyle(.secondary)
                    }
                } else if let loadError {
                    ContentUnavailableView("Aperçu indisponible", systemImage: "exclamationmark.triangle", description: Text(loadError))
                } else {
                    if mode == .rendered {
                        HTMLPreviewWebView(html: html, baseURL: job.artifactURL?.deletingLastPathComponent())
                    } else if mode == .source {
                        SourceTextView(text: markdown)
                    } else {
                        artifactGallery
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()
            HStack {
                Button("Fermer") { dismiss() }
                Spacer()
                if job.stage == .awaitingReview {
                    Button("Écarter", role: .destructive, action: onReject)
                    Button("Approuver", action: onApprove)
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.defaultAction)
                }
            }
            .padding()
        }
        .frame(minWidth: 720, minHeight: 620)
        .task { await loadMarkdown() }
    }

    private func loadMarkdown() async {
        guard let url = job.artifactURL else {
            loadError = "Le fichier Markdown temporaire est introuvable."
            isLoading = false
            return
        }
        do {
            let document = try await MarkdownHTMLRenderer.render(url)
            markdown = document.markdown
            html = document.html
            visualArtifacts = loadArtifacts()
        } catch {
            loadError = error.localizedDescription
        }
        isLoading = false
    }

    @ViewBuilder
    private var artifactGallery: some View {
        if visualArtifacts.isEmpty {
            ContentUnavailableView(
                "Aucun tableau ou figure conservé",
                systemImage: "photo.on.rectangle.angled",
                description: Text("Le document reste consultable en Markdown.")
            )
        } else {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 22) {
                    ForEach(visualArtifacts) { artifact in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(artifact.label).font(.headline)
                                Spacer()
                                if let page = artifact.page { Text("Page \(page)").foregroundStyle(.secondary) }
                            }
                            if let imageURL = artifact.imageURL,
                               let image = NSImage(contentsOf: imageURL) {
                                Image(nsImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxWidth: .infinity)
                                    .clipShape(.rect(cornerRadius: 8))
                            }
                            if !artifact.caption.isEmpty {
                                Text(artifact.caption).font(.callout).foregroundStyle(.secondary)
                            }
                        }
                        .padding(14)
                        .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 12))
                    }
                }
                .padding(20)
            }
        }
    }

    private func loadArtifacts() -> [PreviewArtifact] {
        guard let bundle = job.visualArtifactBundleURL,
              let data = try? Data(contentsOf: bundle.appendingPathComponent("manifest.json")),
              let manifest = try? JSONDecoder().decode(PreviewArtifactManifest.self, from: data) else {
            return []
        }
        return manifest.artifacts.map { item in
            var copy = item
            copy.bundleURL = bundle
            return copy
        }
    }
}

private struct PreviewArtifactManifest: Decodable {
    let artifacts: [PreviewArtifact]
}

private struct PreviewArtifact: Decodable, Identifiable {
    let artifactID: String
    let label: String
    let page: Int?
    let caption: String
    let image: String?
    var bundleURL: URL?

    var id: String { artifactID }
    var imageURL: URL? {
        guard let image, let bundleURL else { return nil }
        return bundleURL.appendingPathComponent(image)
    }

    enum CodingKeys: String, CodingKey {
        case artifactID = "artifact_id"
        case label, page, caption, image
    }
}
