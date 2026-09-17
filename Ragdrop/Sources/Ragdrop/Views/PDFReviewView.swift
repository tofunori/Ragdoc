import PDFKit
import SwiftUI

// Ownership is transferred once from the loader to the main actor. The loader
// never touches this PDFDocument after returning it.
private struct LoadedPDF: @unchecked Sendable { let document: PDFDocument? }

struct PDFReviewView: View {
    let url: URL
    @Binding var page: Int
    @Binding var pageCount: Int
    @Binding var available: Bool
    @State private var document: PDFDocument?
    @State private var loading = true

    var body: some View {
        Group {
            if let document {
                NativePDFView(document: document, page: $page)
            } else if loading {
                ProgressView("Ouverture du PDF…")
            } else {
                ContentUnavailableView("PDF indisponible", systemImage: "doc.questionmark",
                    description: Text("Le fichier original est absent, illisible ou verrouillé. L’extraction reste consultable."))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: url) {
            available = false
            pageCount = 0
            document = nil
            loading = true
            let loaded = await Task.detached(priority: .userInitiated) {
                LoadedPDF(document: PDFDocument(url: url))
            }.value
            guard !Task.isCancelled else { return }
            if let pdf = loaded.document, !pdf.isLocked, pdf.pageCount > 0 {
                document = pdf
                pageCount = pdf.pageCount
                available = true
                page = min(max(1, page), pdf.pageCount)
            }
            loading = false
        }
    }
}

private struct NativePDFView: NSViewRepresentable {
    @Environment(\.colorScheme) private var colorScheme
    let document: PDFDocument
    @Binding var page: Int
    func makeCoordinator() -> Coordinator { Coordinator(page: $page) }
    func makeNSView(context: Context) -> PDFView {
        let view = PDFView()
        view.autoScales = true
        view.backgroundColor = RagdropTheme.nsCanvas
        view.displayMode = .singlePage
        view.document = document
        context.coordinator.view = view
        NotificationCenter.default.addObserver(context.coordinator, selector: #selector(Coordinator.changed),
                                               name: .PDFViewPageChanged, object: view)
        return view
    }
    func updateNSView(_ view: PDFView, context: Context) {
        view.appearance = NSAppearance(named: colorScheme == .dark ? .darkAqua : .aqua)
        view.backgroundColor = RagdropTheme.nsCanvas
        context.coordinator.page = $page
        if view.document !== document { view.document = document }
        guard let target = document.page(at: page - 1), view.currentPage !== target else { return }
        view.go(to: target)
    }
    static func dismantleNSView(_ view: PDFView, coordinator: Coordinator) {
        NotificationCenter.default.removeObserver(coordinator)
        coordinator.view = nil
    }
    @MainActor final class Coordinator: NSObject {
        var page: Binding<Int>
        weak var view: PDFView?
        init(page: Binding<Int>) { self.page = page }
        @objc func changed() {
            guard let view, let current = view.currentPage, let document = view.document else { return }
            let number = document.index(for: current) + 1
            // PDFKit can notify while SwiftUI is updating the represented view.
            DispatchQueue.main.async { [weak self] in
                guard let self, self.view != nil, self.page.wrappedValue != number else { return }
                self.page.wrappedValue = number
            }
        }
    }
}
