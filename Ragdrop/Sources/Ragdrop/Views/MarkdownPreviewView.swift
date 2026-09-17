import AppKit
import SwiftUI

struct MarkdownPreviewView: View {
    let job: ImportJob
    var canDecide = true
    var remainingReviews = 0
    var onNext: (() -> Void)?
    let onApprove: () -> Void
    let onReject: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var document: ReviewDocument?
    @State private var loadError: String?
    @State private var isLoading = true
    @State private var mode = PreviewMode.rendered
    @State private var page = 1
    @State private var pageCount = 0
    @State private var pdfAvailable = false
    @State private var pageOnly = false
    @State private var visibleText = ""
    @State private var html = ""
    @State private var rendering = false
    @State private var renderError: String?
    @State private var truncated = false
    @State private var showRejectConfirmation = false

    private enum PreviewMode: String, CaseIterable, Identifiable {
        case rendered = "Rendu"
        case source = "Source"
        case visuals = "Tableaux et figures"
        var id: Self { self }
    }
    private var renderKey: String { "\(document != nil)-\(pageOnly)-\(pageOnly ? page : 0)" }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(job.displayName).font(.headline).lineLimit(1).help(job.displayName)
                    Text("Révision de l’extraction · \(document?.converter ?? job.converterName ?? "Convertisseur non renseigné")")
                        .font(.caption).foregroundStyle(RagdropTheme.secondary)
                }
                Spacer()
                if remainingReviews > 0 { Text("\(remainingReviews) à réviser").font(.caption).foregroundStyle(RagdropTheme.secondary) }
                if let onNext { Button("Suivant", action: onNext) }
            }
            .padding(18)
            Divider()
            HSplitView {
                VStack(spacing: 0) {
                    HStack {
                        Label("PDF original", systemImage: "doc.richtext").font(.subheadline.weight(.semibold))
                        Spacer()
                        Button { page -= 1 } label: { Image(systemName: "chevron.left") }
                            .disabled(page <= 1 || !pdfAvailable).help("Page précédente")
                        Text(pageCount > 0 ? "\(page) / \(pageCount)" : "—").monospacedDigit()
                        Button { page += 1 } label: { Image(systemName: "chevron.right") }
                            .disabled(page >= pageCount || !pdfAvailable).help("Page suivante")
                    }.padding(12)
                    Divider()
                    PDFReviewView(url: job.fileURL, page: $page, pageCount: $pageCount, available: $pdfAvailable)
                }
                .frame(minWidth: 300, idealWidth: 510, maxWidth: .infinity)
                VStack(spacing: 0) {
                    Picker("Extraction", selection: $mode) {
                        ForEach(PreviewMode.allCases) { Text($0.rawValue).tag($0) }
                    }.pickerStyle(.segmented).padding(12)
                    if let document {
                        VStack(alignment: .leading, spacing: 5) {
                            if !document.spans.isEmpty {
                                Toggle("Extrait lié à la page \(page)", isOn: $pageOnly)
                                    .toggleStyle(.checkbox).disabled(mode == .visuals)
                            }
                            Text(document.provenanceNote).font(.caption).foregroundStyle(RagdropTheme.secondary)
                            if truncated {
                                Text("Aperçu limité aux 200 000 premiers caractères. Consultez le fichier complet ou choisissez une page.")
                                    .font(.caption).foregroundStyle(RagdropTheme.warning)
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 12).padding(.bottom, 10)
                    }
                    Divider()
                    extraction.frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                .frame(minWidth: 390, idealWidth: 560, maxWidth: .infinity)
            }
            Divider()
            HStack {
                Button("Fermer") { dismiss() }.keyboardShortcut(.cancelAction)
                if let url = job.artifactURL {
                    Button("Fichier extrait") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                }
                Spacer()
                if job.stage == .awaitingReview {
                    if !canDecide { Text("Décision disponible à la fin du traitement.").font(.caption).foregroundStyle(RagdropTheme.secondary) }
                    Button("Écarter…", role: .destructive) { showRejectConfirmation = true }
                        .disabled(!canDecide)
                    Button("Approuver l’extraction", action: onApprove)
                        .buttonStyle(RagdropPrimaryButtonStyle())
                        .disabled(!canDecide || document == nil || !pdfAvailable)
                        .help("Autorise l’envoi à Ragdoc; l’ajout sera lancé depuis la file.")
                }
            }.padding()
        }
        .frame(minWidth: 780, idealWidth: 1120, minHeight: 600, idealHeight: 800)
        .ragdropSurface()
        .confirmationDialog("Écarter cette extraction ? Le PDF original est conservé.", isPresented: $showRejectConfirmation) {
            Button("Écarter l’extraction", role: .destructive, action: onReject)
        }
        .task(id: job.id) {
            do {
                let loaded = try await ReviewDocument.load(job)
                guard !Task.isCancelled else { return }
                document = loaded
                pageOnly = !loaded.spans.isEmpty
            } catch { loadError = error.localizedDescription }
            isLoading = false
        }
        .task(id: renderKey) { await renderSelection() }
    }

    @ViewBuilder private var extraction: some View {
        if isLoading {
            ProgressView("Lecture de l’extraction…")
        } else if let loadError {
            ContentUnavailableView("Extraction indisponible", systemImage: "exclamationmark.triangle", description: Text(loadError))
        } else if mode == .visuals {
            if document?.artifacts.isEmpty != false {
                ContentUnavailableView("Aucun élément visuel conservé", systemImage: "photo.on.rectangle",
                                       description: Text("Consultez le rendu ou la source pour examiner les tableaux intégrés au texte."))
            } else {
                ScrollView {
                    LazyVStack(spacing: 18) {
                        ForEach(document?.artifacts ?? []) { artifact in
                            ReviewArtifactView(artifact: artifact, bundleURL: job.visualArtifactBundleURL,
                                canNavigate: pdfAvailable && !((document?.spans.isEmpty) ?? true)
                                    && (artifact.page ?? 0) > 0 && (artifact.page ?? 0) <= pageCount,
                                onPage: { if let target = artifact.page { page = target } })
                        }
                    }.padding(16)
                }
            }
        } else if pageOnly && document?.spans.contains(where: { $0.page == page }) != true {
            ContentUnavailableView("Aucun extrait relié à cette page", systemImage: "text.page",
                                   description: Text("Affichez le document complet pour comparer manuellement."))
        } else if mode == .source {
            SourceTextView(text: visibleText)
        } else if rendering {
            ProgressView("Préparation du rendu…")
        } else if let renderError {
            ContentUnavailableView("Rendu indisponible", systemImage: "exclamationmark.triangle",
                                   description: Text("\(renderError) L’onglet Source reste disponible."))
        } else {
            HTMLPreviewWebView(html: html, baseURL: job.artifactURL?.deletingLastPathComponent())
        }
    }

    private func renderSelection() async {
        guard let document else { return }
        rendering = true
        renderError = nil
        let selectedPage = pageOnly ? page : nil
        let selection = await Task.detached(priority: .userInitiated) {
            let text = selectedPage.flatMap { document.excerpt(page: $0) } ?? (selectedPage == nil ? document.markdown : "")
            return (String(text.prefix(200_000)), text.count > 200_000)
        }.value
        guard !Task.isCancelled else { return }
        visibleText = selection.0
        truncated = selection.1
        do {
            let rendered = try await MarkdownHTMLRenderer.render(text: selection.0)
            guard !Task.isCancelled else { return }
            html = rendered.html
        } catch {
            guard !Task.isCancelled else { return }
            renderError = error.localizedDescription
        }
        rendering = false
    }
}
