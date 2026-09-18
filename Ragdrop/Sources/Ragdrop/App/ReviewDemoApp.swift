#if RAGDROP_REVIEW_DEMO || RAGDROP_THEME_DEMO
import AppKit
import CryptoKit
import PDFKit
import SwiftUI

#if RAGDROP_REVIEW_DEMO
/// This build has no ImportStore, settings scene, network action or user queue.
@main
struct ReviewDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    var body: some Scene {
        WindowGroup("Ragdrop — Isolated demo") { ReviewDemoView() }
            .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unifiedCompact(showsTitle: false))
        .defaultSize(width: 1180, height: 900)
    }
}

private struct ReviewDemoView: View {
    @State private var jobs: [ImportJob] = []
    @State private var selected = 0
    @State private var notice = "Synthetic data · no transfer or indexing"
    @State private var fixtureError: String?
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Isolated demo", systemImage: "testtube.2").font(.headline)
                Picker("Scenario", selection: $selected) {
                    Text("Linked pages").tag(0)
                    Text("Legacy document").tag(1)
                    Text("Missing PDF").tag(2)
                    Text("Long extraction").tag(3)
                }.frame(width: 300)
                Spacer()
                Text(notice).font(.caption).foregroundStyle(.secondary)
            }.padding(12)
            if !jobs.isEmpty {
                BatchProgressView(jobs: jobs, message: "Compare the files before deciding.").padding(.horizontal, 12)
                MarkdownPreviewView(job: jobs[selected], remainingReviews: jobs.filter { $0.stage == .awaitingReview }.count,
                    onNext: { selected = (selected + 1) % jobs.count },
                    onApprove: { jobs[selected].stage = .readyForIndexing; notice = "Simulated approval — no transfer" },
                    onReject: { jobs[selected].stage = .rejected; notice = "Simulated rejection — no file deleted" })
                    .id(jobs[selected].id)
            } else if let fixtureError {
                Text(fixtureError).padding()
            }
        }.task {
            do { jobs = try ReviewDemoFixtures.make() }
            catch { fixtureError = error.localizedDescription }
        }
    }
}

#endif

enum ReviewDemoFixtures {
    @MainActor static func make() throws -> [ImportJob] {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("ragdrop-demo-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let pdfURL = directory.appendingPathComponent("Glacier — synthetic example.pdf")
        let pdf = PDFDocument()
        for index in 1...3 {
            let image = NSImage(size: NSSize(width: 595, height: 842))
            image.lockFocus()
            NSColor.white.setFill()
            NSRect(x: 0, y: 0, width: 595, height: 842).fill()
            let title = "SCIENTIFIC REVIEW\n\nSynthetic example · page \(index)"
            title.draw(in: NSRect(x: 45, y: 650, width: 510, height: 140), withAttributes: [
                .font: NSFont.systemFont(ofSize: 23, weight: .semibold), .foregroundColor: NSColor.black])
            let body = index == 2
                ? "Table 1 — Fictional values\n\nSite A       0.72\nSite B       0.48\n\nThese values are only for testing the interface."
                : "This page compares the original PDF with its extraction.\n\nCheck numbers, units, symbols and references.\n\nIndexing confirms that passages are stored. It does not guarantee scientific accuracy."
            body.draw(in: NSRect(x: 45, y: 310, width: 490, height: 290), withAttributes: [
                .font: NSFont.systemFont(ofSize: 18), .foregroundColor: NSColor.black])
            image.unlockFocus()
            if let page = PDFPage(image: image) { pdf.insert(page, at: index - 1) }
        }
        guard pdf.write(to: pdfURL) else { throw CocoaError(.fileWriteUnknown) }
        let sections = [
            "# Scientific review\n\nSynthetic example · page 1\n\nCompare **numbers**, units and references. Accents: e\u{301}.\n\n",
            "## Table 1 — Fictional values\n\n| Site | Albedo |\n|---|---|\n| A | 0.72 |\n| B | 0.48 |\n\nThese values are only for testing the interface.\n\n",
            "## Passage verification\n\nIndexing confirms that passages are stored. It does not guarantee scientific accuracy.\n"
        ]
        let markdown = sections.joined()
        let textURL = directory.appendingPathComponent("extraction.md")
        try markdown.write(to: textURL, atomically: true, encoding: .utf8)
        var offset = 0
        let spans = sections.enumerated().map { index, text -> [String: Int] in
            let start = offset
            offset += text.unicodeScalars.count
            return ["page": index + 1, "start": start, "end": offset]
        }
        func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        let sidecar: [String: Any] = ["content_sha256": hash(Data(markdown.utf8)),
            "parsed_pdf_sha256": hash(try Data(contentsOf: pdfURL)), "page_spans": spans, "parser": "synthetic-demo"]
        let metadataURL = directory.appendingPathComponent("metadata.json")
        try JSONSerialization.data(withJSONObject: sidecar).write(to: metadataURL)
        let bundle = directory.appendingPathComponent("visuals")
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        let manifest: [String: Any] = ["artifacts": [["artifact_id": "demo-table", "label": "Table 1", "page": 2,
            "caption": "Fictional values for testing", "body": sections[1], "image": "missing-table.png"]]]
        try JSONSerialization.data(withJSONObject: manifest).write(to: bundle.appendingPathComponent("manifest.json"))
        var linked = ImportJob(fileURL: pdfURL)
        linked.stage = .awaitingReview
        linked.artifactURL = textURL
        linked.metadataURL = metadataURL
        linked.visualArtifactBundleURL = bundle
        linked.converterName = "Demo"
        var legacy = ImportJob(fileURL: pdfURL)
        legacy.stage = .awaitingReview
        legacy.artifactURL = textURL
        var missing = ImportJob(fileURL: directory.appendingPathComponent("Missing PDF.pdf"))
        missing.stage = .awaitingReview
        missing.artifactURL = textURL
        let longURL = directory.appendingPathComponent("long.md")
        try String(repeating: markdown, count: 600).write(to: longURL, atomically: true, encoding: .utf8)
        var long = ImportJob(fileURL: pdfURL)
        long.stage = .awaitingReview
        long.artifactURL = longURL
        return [linked, legacy, missing, long]
    }
}
#endif
