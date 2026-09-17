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
        WindowGroup("Ragdrop — Validation isolée") { ReviewDemoView() }
            .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unifiedCompact(showsTitle: false))
        .defaultSize(width: 1180, height: 900)
    }
}

private struct ReviewDemoView: View {
    @State private var jobs: [ImportJob] = []
    @State private var selected = 0
    @State private var notice = "Données synthétiques · aucun transfert ni indexation"
    @State private var fixtureError: String?
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Validation isolée", systemImage: "testtube.2").font(.headline)
                Picker("Scénario", selection: $selected) {
                    Text("Pages liées").tag(0)
                    Text("Ancien document").tag(1)
                    Text("PDF manquant").tag(2)
                    Text("Longue extraction").tag(3)
                }.frame(width: 300)
                Spacer()
                Text(notice).font(.caption).foregroundStyle(.secondary)
            }.padding(12)
            if !jobs.isEmpty {
                BatchProgressView(jobs: jobs, message: "Comparez les fichiers avant de décider.").padding(.horizontal, 12)
                MarkdownPreviewView(job: jobs[selected], remainingReviews: jobs.filter { $0.stage == .awaitingReview }.count,
                    onNext: { selected = (selected + 1) % jobs.count },
                    onApprove: { jobs[selected].stage = .readyForIndexing; notice = "Approbation simulée — aucun envoi" },
                    onReject: { jobs[selected].stage = .rejected; notice = "Rejet simulé — aucun fichier supprimé" })
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
        let pdfURL = directory.appendingPathComponent("Glacier — exemple synthétique.pdf")
        let pdf = PDFDocument()
        for index in 1...3 {
            let image = NSImage(size: NSSize(width: 595, height: 842))
            image.lockFocus()
            NSColor.white.setFill()
            NSRect(x: 0, y: 0, width: 595, height: 842).fill()
            let title = "RÉVISION SCIENTIFIQUE\n\nExemple synthétique · page \(index)"
            title.draw(in: NSRect(x: 45, y: 650, width: 510, height: 140), withAttributes: [
                .font: NSFont.systemFont(ofSize: 23, weight: .semibold), .foregroundColor: NSColor.black])
            let body = index == 2
                ? "Tableau 1 — Valeurs fictives\n\nSite A       0,72\nSite B       0,48\n\nLes valeurs servent uniquement à tester l’interface."
                : "Cette page sert à comparer le PDF original et son extraction.\n\nVérifier les nombres, unités, symboles et références.\n\nL’indexation confirme la présence de passages. Elle ne garantit pas leur exactitude scientifique."
            body.draw(in: NSRect(x: 45, y: 310, width: 490, height: 290), withAttributes: [
                .font: NSFont.systemFont(ofSize: 18), .foregroundColor: NSColor.black])
            image.unlockFocus()
            if let page = PDFPage(image: image) { pdf.insert(page, at: index - 1) }
        }
        guard pdf.write(to: pdfURL) else { throw CocoaError(.fileWriteUnknown) }
        let sections = [
            "# Révision scientifique\n\nExemple synthétique · page 1\n\nComparer les **nombres**, unités et références. Accents : e\u{301}.\n\n",
            "## Tableau 1 — Valeurs fictives\n\n| Site | Albédo |\n|---|---|\n| A | 0,72 |\n| B | 0,48 |\n\nCes valeurs servent uniquement à tester l’interface.\n\n",
            "## Contrôle des passages\n\nL’indexation confirme la présence de passages. Elle ne garantit pas leur exactitude scientifique.\n"
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
        let manifest: [String: Any] = ["artifacts": [["artifact_id": "demo-table", "label": "Tableau 1", "page": 2,
            "caption": "Valeurs fictives pour validation", "body": sections[1], "image": "missing-table.png"]]]
        try JSONSerialization.data(withJSONObject: manifest).write(to: bundle.appendingPathComponent("manifest.json"))
        var linked = ImportJob(fileURL: pdfURL)
        linked.stage = .awaitingReview
        linked.artifactURL = textURL
        linked.metadataURL = metadataURL
        linked.visualArtifactBundleURL = bundle
        linked.converterName = "Démonstration"
        var legacy = ImportJob(fileURL: pdfURL)
        legacy.stage = .awaitingReview
        legacy.artifactURL = textURL
        var missing = ImportJob(fileURL: directory.appendingPathComponent("PDF absent.pdf"))
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
