import CryptoKit
import Foundation

struct ReviewPageSpan: Decodable, Sendable {
    let page: Int
    let start: Int
    let end: Int
}

struct ReviewArtifact: Decodable, Identifiable, Sendable {
    let artifactID: String
    let label: String
    let page: Int?
    let caption: String?
    let body: String?
    let image: String?
    var id: String { artifactID }
    enum CodingKeys: String, CodingKey {
        case artifactID = "artifact_id"
        case label, page, caption, body, image
    }
}

struct ReviewDocument: Sendable {
    let markdown: String
    let spans: [ReviewPageSpan]
    let artifacts: [ReviewArtifact]
    let provenanceNote: String
    let converter: String?

    /// Python offsets count Unicode scalars, not Swift extended grapheme clusters.
    func excerpt(page: Int) -> String? {
        let scalars = Array(markdown.unicodeScalars)
        let matches = spans.filter { $0.page == page }
        guard !matches.isEmpty else { return nil }
        return matches.map { String(String.UnicodeScalarView(scalars[$0.start..<$0.end])) }
            .joined(separator: "\n\n")
    }

    static func validSpans(_ spans: [ReviewPageSpan], scalarCount: Int) -> [ReviewPageSpan] {
        let ordered = spans.sorted { $0.start < $1.start }
        var previousEnd = 0
        for span in ordered {
            guard span.page > 0, span.start >= previousEnd, span.end > span.start,
                  span.end <= scalarCount else { return [] }
            previousEnd = span.end
        }
        return ordered
    }

    static func load(_ job: ImportJob) async throws -> ReviewDocument {
        try await Task.detached(priority: .userInitiated) {
            guard let url = job.artifactURL else { throw MarkdownPreviewError.unreadable }
            let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            guard size <= 20_000_000 else { throw ReviewLoadError.tooLarge }
            let data = try Data(contentsOf: url)
            guard let markdown = String(data: data, encoding: .utf8) else { throw MarkdownPreviewError.unreadable }
            let sidecar = job.metadataURL.flatMap { try? Data(contentsOf: $0) }
                .flatMap { try? JSONDecoder().decode(Sidecar.self, from: $0) }
            let manifest = job.visualArtifactBundleURL.flatMap {
                try? Data(contentsOf: $0.appendingPathComponent("manifest.json"))
            }.flatMap { try? JSONDecoder().decode(Manifest.self, from: $0) }
            let textHash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            let pdfHash = try? hashFile(job.fileURL)
            let matches = sidecar?.content_sha256 == textHash && pdfHash != nil
                && sidecar?.parsed_pdf_sha256 == pdfHash
            let spans = matches ? validSpans(sidecar?.page_spans ?? [], scalarCount: markdown.unicodeScalars.count) : []
            return ReviewDocument(
                markdown: markdown, spans: spans, artifacts: manifest?.artifacts ?? [],
                provenanceNote: spans.isEmpty
                    ? "Correspondance de pages indisponible ou non vérifiable. Comparaison manuelle."
                    : "Repères du convertisseur liés à ces fichiers. Ils peuvent ne couvrir qu’une partie de la page.",
                converter: job.converterName ?? sidecar?.parser
            )
        }.value
    }

    private static func hashFile(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private struct Sidecar: Decodable {
        let content_sha256: String?
        let parsed_pdf_sha256: String?
        let page_spans: [ReviewPageSpan]?
        let parser: String?
    }
    private struct Manifest: Decodable { let artifacts: [ReviewArtifact] }
}

enum ReviewLoadError: LocalizedError {
    case tooLarge
    var errorDescription: String? { "L’extraction dépasse 20 Mo. Ouvrez le fichier dans un éditeur externe pour la réviser." }
}
