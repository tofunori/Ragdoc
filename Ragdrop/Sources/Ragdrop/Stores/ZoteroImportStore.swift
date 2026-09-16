import Foundation
import Observation

@MainActor
@Observable
final class ZoteroImportStore {
    var documents: [ZoteroPDF] = []
    var selectedKeys: Set<String> = []
    var isLoading = false
    var errorMessage: String?

    func load() async {
        guard !isLoading else { return }
        isLoading = true
        errorMessage = nil
        do {
            async let zoteroPDFs = ZoteroLibraryService().fetchPDFs()
            async let history = RagdocHistoryService(configuration: .current()).fetchDocuments()
            let (pdfs, indexedDocuments) = try await (zoteroPDFs, history)
            let sources = indexedDocuments.map(\.source)
            documents = pdfs.map { pdf in
                var copy = pdf
                copy.isIndexed = sources.contains { source in
                    source.hasSuffix("_\(pdf.attachmentKey).md")
                        || pdf.fingerprintPrefix.map { prefix in
                            source.hasSuffix("_\(prefix).md")
                        } == true
                }
                return copy
            }
            selectedKeys.formIntersection(Set(documents.filter { !$0.isIndexed }.map(\.attachmentKey)))
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    func setSelected(_ selected: Bool, key: String) {
        if selected { selectedKeys.insert(key) } else { selectedKeys.remove(key) }
    }

    func select(_ documents: [ZoteroPDF]) {
        selectedKeys.formUnion(documents.filter { !$0.isIndexed }.map(\.attachmentKey))
    }

    func clearSelection() {
        selectedKeys.removeAll()
    }

    var selectedURLs: [URL] {
        documents.filter { selectedKeys.contains($0.attachmentKey) }.map(\.fileURL)
    }

    var selectedDocuments: [ZoteroPDF] {
        documents.filter { selectedKeys.contains($0.attachmentKey) }
    }
}
