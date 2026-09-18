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
        case rendered = "Rendered"
        case source = "Source"
        case visuals = "Tables and figures"
        var id: Self { self }
    }
    private var renderKey: String { "\(document != nil)-\(pageOnly)-\(pageOnly ? page : 0)" }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(job.displayName).font(.headline).lineLimit(1).help(job.displayName)
                    Text("Extraction review · \(document?.converter ?? job.converterName ?? "Converter not specified")")
                        .font(.caption).foregroundStyle(RagdropTheme.secondary)
                }
                Spacer()
                if remainingReviews > 0 { Text("\(remainingReviews) to review").font(.caption).foregroundStyle(RagdropTheme.secondary) }
                if let onNext { Button("Next", action: onNext) }
            }
            .padding(18)
            Divider()
            HSplitView {
                VStack(spacing: 0) {
                    HStack {
                        Label("Original PDF", systemImage: "doc.richtext").font(.subheadline.weight(.semibold))
                        Spacer()
                        Button { page -= 1 } label: { Image(systemName: "chevron.left") }
                            .disabled(page <= 1 || !pdfAvailable).help("Previous page")
                        Text(pageCount > 0 ? "\(page) / \(pageCount)" : "—").monospacedDigit()
                        Button { page += 1 } label: { Image(systemName: "chevron.right") }
                            .disabled(page >= pageCount || !pdfAvailable).help("Next page")
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
                                Toggle("Excerpt linked to page \(page)", isOn: $pageOnly)
                                    .toggleStyle(.checkbox).disabled(mode == .visuals)
                            }
                            Text(document.provenanceNote).font(.caption).foregroundStyle(RagdropTheme.secondary)
                            if truncated {
                                Text("Preview limited to the first 200,000 characters. Open the full file or select a page.")
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
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                if let url = job.artifactURL {
                    Button("Extracted file") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                }
                Spacer()
                if job.stage == .awaitingReview {
                    if !canDecide { Text("Review actions are available when processing finishes.").font(.caption).foregroundStyle(RagdropTheme.secondary) }
                    Button("Reject…", role: .destructive) { showRejectConfirmation = true }
                        .disabled(!canDecide)
                    Button("Approve extraction", action: onApprove)
                        .buttonStyle(RagdropPrimaryButtonStyle())
                        .disabled(!canDecide || document == nil || !pdfAvailable)
                        .help("Approves transfer to Ragdoc; start adding from the queue.")
                }
            }.padding()
        }
        .frame(minWidth: 780, idealWidth: 1120, minHeight: 600, idealHeight: 800)
        .ragdropSurface()
        .confirmationDialog("Reject this extraction? The original PDF will be kept.", isPresented: $showRejectConfirmation) {
            Button("Reject extraction", role: .destructive, action: onReject)
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
            ProgressView("Reading extraction…")
        } else if let loadError {
            ContentUnavailableView("Extraction unavailable", systemImage: "exclamationmark.triangle", description: Text(loadError))
        } else if mode == .visuals {
            if document?.artifacts.isEmpty != false {
                ContentUnavailableView("No visual items saved", systemImage: "photo.on.rectangle",
                                       description: Text("Use the rendered view or source to inspect tables embedded in the text."))
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
            ContentUnavailableView("No excerpt linked to this page", systemImage: "text.page",
                                   description: Text("Show the full document to compare manually."))
        } else if mode == .source {
            SourceTextView(text: visibleText)
        } else if rendering {
            ProgressView("Preparing rendered view…")
        } else if let renderError {
            ContentUnavailableView("Rendered view unavailable", systemImage: "exclamationmark.triangle",
                                   description: Text("\(renderError) The Source tab is still available."))
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
