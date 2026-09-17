import Foundation
import Observation

@MainActor
@Observable
final class ZoteroImportStore {
    let isIsolated: Bool
    let allowedKeys: Set<String>?
    let excludedJobs: [ImportJob]
    init(isIsolated: Bool = false, allowedKeys: Set<String>? = nil, excludedJobs: [ImportJob] = []) {
        self.isIsolated = isIsolated
        self.allowedKeys = allowedKeys
        self.excludedJobs = excludedJobs
    }
    var documents: [ZoteroPDF] = []
    var selectedKeys: Set<String> = []
    var isLoading = false
    var errorMessage: String?

    func load() async {
        guard !isIsolated else { return }
        guard !isLoading else { return }
        isLoading = true
        errorMessage = nil
        do {
            async let zoteroPDFs = ZoteroLibraryService().fetchPDFs(onlyKeys: allowedKeys)
            async let history = RagdocHistoryService(configuration: .current()).fetchDocuments(includePDFIdentity: true)
            let (pdfs, indexedDocuments) = try await (zoteroPDFs, history)
            let queueHashes = try await ZoteroFingerprintCache().hashes(for: excludedJobs)
            documents = pdfs.filter { document in
                !ZoteroDeduplication.isQueued(document, jobs: excludedJobs)
                    && !(document.fingerprint.map { queueHashes.contains($0) } ?? false)
            }.map { pdf in
                var copy = pdf
                copy.isIndexed = ZoteroDeduplication.isIndexed(pdf, history: indexedDocuments)
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
