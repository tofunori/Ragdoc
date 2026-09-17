import CryptoKit
import Foundation
import Testing
@testable import Ragdrop

struct ReviewDocumentTests {
    @Test func pythonOffsetsPreserveCombiningMarksAndEmoji() {
        let text = "e\u{301} 🌨️\nPage deux"
        let start = "e\u{301} 🌨️\n".unicodeScalars.count
        let document = ReviewDocument(markdown: text,
            spans: [.init(page: 2, start: start, end: text.unicodeScalars.count)],
            artifacts: [], provenanceNote: "", converter: nil)
        #expect(document.excerpt(page: 2) == "Page deux")
        #expect(document.excerpt(page: 1) == nil)
    }

    @Test func malformedOrOverlappingSpansDisablePageLinking() {
        for spans: [ReviewPageSpan] in [
            [.init(page: 0, start: 0, end: 2)],
            [.init(page: 1, start: -1, end: 2)],
            [.init(page: 1, start: 0, end: 11)],
            [.init(page: 1, start: 4, end: 4)],
            [.init(page: 1, start: 0, end: 5), .init(page: 2, start: 4, end: 8)]
        ] {
            #expect(ReviewDocument.validSpans(spans, scalarCount: 10).isEmpty)
        }
    }

    @Test func provenanceRequiresBothMatchingFilesAndSupportsLegacy() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let pdf = folder.appendingPathComponent("test.pdf")
        let markdown = folder.appendingPathComponent("test.md")
        let metadata = folder.appendingPathComponent("test.metadata.json")
        let pdfData = Data("synthetic PDF bytes".utf8)
        let textData = Data("# Page 1\nTexte".utf8)
        try pdfData.write(to: pdf)
        try textData.write(to: markdown)
        var job = ImportJob(fileURL: pdf)
        job.artifactURL = markdown
        #expect(try await ReviewDocument.load(job).spans.isEmpty)
        let contentHash = SHA256.hash(data: textData).map { String(format: "%02x", $0) }.joined()
        let pdfHash = SHA256.hash(data: pdfData).map { String(format: "%02x", $0) }.joined()
        let sidecar: [String: Any] = ["content_sha256": contentHash, "parsed_pdf_sha256": pdfHash,
            "parser": "mistral-ocr", "page_spans": [["page": 1, "start": 0, "end": 14]]]
        try JSONSerialization.data(withJSONObject: sidecar).write(to: metadata)
        job.metadataURL = metadata
        #expect(try await ReviewDocument.load(job).spans.count == 1)
        try Data("modified PDF".utf8).write(to: pdf)
        #expect(try await ReviewDocument.load(job).spans.isEmpty)
        try pdfData.write(to: pdf)
        try Data("changed markdown".utf8).write(to: markdown)
        #expect(try await ReviewDocument.load(job).spans.isEmpty)
        try FileManager.default.removeItem(at: pdf)
        #expect(try await ReviewDocument.load(job).spans.isEmpty)
    }

    @Test func oldQueueDoesNotRequireConverterName() throws {
        var job = ImportJob(fileURL: URL(fileURLWithPath: "/tmp/legacy.pdf"))
        job.stage = .awaitingReview
        let data = try JSONEncoder().encode(job)
        let decoded = try JSONDecoder().decode(ImportJob.self, from: data)
        #expect(decoded.converterName == nil)
        job.converterName = "MinerU"
        #expect(try JSONDecoder().decode(ImportJob.self, from: JSONEncoder().encode(job)).converterName == "MinerU")
    }

    @Test func countsDoNotTurnDuplicatesRejectionsOrRetriesIntoSuccesses() {
        let stages: [ImportStage] = [.completed, .duplicate, .rejected, .failed, .awaitingReview, .readyForIndexing, .indexing]
        var jobs = stages.map { stage in
            var job = ImportJob(fileURL: URL(fileURLWithPath: "/tmp/test.pdf"))
            job.stage = stage
            return job
        }
        jobs[5].errorDetails = "Indexation à reprendre"
        let summary = BatchSummary(jobs: jobs)
        #expect(summary.added == 1)
        #expect(summary.review == 1)
        #expect(summary.errors == 2)
        #expect(summary.active == 1)
    }
}
