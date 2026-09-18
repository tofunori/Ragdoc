import Foundation
import Darwin
import Testing
@testable import Ragdrop

struct LocalLibraryTests {
    @Test func freshInstallUsesLocalButExistingServerIsPreserved() {
        let name = "ragdrop-location-test-\(UUID().uuidString)"
        let prefs = UserDefaults(suiteName: name)!
        defer { prefs.removePersistentDomain(forName: name) }
        #expect(LibraryLocation.resolve(defaults: prefs, hasLegacyQueue: false) == .local)
        #expect(LibraryLocation.resolve(defaults: prefs, hasLegacyQueue: true) == .server)
        prefs.set("/custom/converter.py", forKey: "converterPath")
        #expect(LibraryLocation.resolve(defaults: prefs, hasLegacyQueue: false) == .server)
        prefs.set("local", forKey: "libraryLocation")
        #expect(LibraryLocation.resolve(defaults: prefs, hasLegacyQueue: true) == .local)
        prefs.set("server", forKey: "libraryLocation")
        #expect(LibraryLocation.resolve(defaults: prefs, hasLegacyQueue: false) == .server)
    }
    @Test func localConfigurationSupportsSpacesWithoutRelaxingSSHValidation() throws {
        let connection = LocalConnection(version: 1, python: "/tmp/python", script: "/tmp/bridge.py", library: "/tmp/My papers – été")
        let local = PipelineConfiguration(converterPath: "/tmp/convert.py", nasHost: "", remoteRoot: connection.library, location: .local, localConnection: connection)
        try local.validate()
        let remote = PipelineConfiguration(converterPath: "", nasHost: "bad;host", remoteRoot: "/srv/ragdoc")
        #expect(throws: PipelineError.self) { try remote.validate() }
        let unprepared = PipelineConfiguration(converterPath: "", nasHost: "", remoteRoot: "/tmp/library", location: .local)
        #expect(throws: PipelineError.self) { try unprepared.validate() }
    }
    @Test func reviewedTransferPreservesSourcesAndRejectsTraversal() throws {
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temp) }
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        let markdown = temp.appendingPathComponent("source.md")
        let metadata = temp.appendingPathComponent("source.metadata.json")
        try "# Original content".write(to: markdown, atomically: true, encoding: .utf8)
        try "{\"review_status\":\"approved\"}".write(to: metadata, atomically: true, encoding: .utf8)
        let root = temp.appendingPathComponent("Library with spaces")
        let artifact = ConversionArtifact(sourceURL: temp.appendingPathComponent("original.pdf"), markdownURL: markdown, remoteFilename: "paper_abcdef123456.md", metadataURL: metadata, artifactBundleURL: nil, artifactCount: 0)
        try LocalTransfer.save(artifact, root: root)
        #expect(try Data(contentsOf: root.appendingPathComponent("articles_markdown/paper_abcdef123456.md")) == Data(contentsOf: markdown))
        #expect(try Data(contentsOf: root.appendingPathComponent("articles_markdown/paper_abcdef123456.metadata.json")) == Data(contentsOf: metadata))
        let malicious = ConversionArtifact(sourceURL: artifact.sourceURL, markdownURL: markdown, remoteFilename: "../escaped.md", metadataURL: nil, artifactBundleURL: nil, artifactCount: 0)
        #expect(throws: PipelineError.self) { try LocalTransfer.save(malicious, root: root) }
    }
    @Test func processCancellationStopsAChild() async throws {
        let task = Task { try await ProcessRunner.run(executable: "/bin/sleep", arguments: ["30"], label: "fixture", timeout: 5) }
        try await Task.sleep(for: .milliseconds(150)); task.cancel()
        do { _ = try await task.value; Issue.record("Canceled child unexpectedly succeeded") }
        catch is CancellationError {}
        catch { Issue.record("Unexpected error: \(error)") }
    }
    @Test func missingEngineIsActionableAndDoesNotPublishConnection() async throws {
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let engine = LocalEngine(support: temp, resources: temp.appendingPathComponent("missing"))
        do { _ = try await engine.prepare(library: temp.appendingPathComponent("library")) { _ in }; Issue.record("Missing engine accepted") }
        catch { #expect(error.localizedDescription.contains("full Ragdrop app")) }
        #expect(!FileManager.default.fileExists(atPath: engine.connectionURL.path))
    }
    @MainActor @Test func brokenRuntimeRetainsTheCustomLibrary() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let connection = LocalConnection(version: 1, python: root.appendingPathComponent("missing-python").path,
            script: root.appendingPathComponent("missing.py").path, library: root.appendingPathComponent("My existing papers").path)
        try JSONEncoder().encode(connection).write(to: root.appendingPathComponent("local-connection.json"))
        let engine = LocalEngine(support: root, resources: root)
        #expect(engine.connection == nil)
        #expect(engine.recordedConnection?.library == connection.library)
        let setup = LocalSetupStore(engine: engine)
        #expect(setup.library.path == connection.library)
        #expect(!setup.isReady)
    }
    @MainActor @Test func validationCannotStartPaidIndexRecovery() {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let setup = LocalSetupStore(engine: LocalEngine(support: root, resources: root), allowsPersonalActions: false)
        setup.isReady = true
        setup.rebuildIndex()
        #expect(!setup.isPreparing)
        #expect(setup.progress.isEmpty)
    }
    @Test func cancellationStopsAChildAndItsDescendant() async throws {
        let marker = FileManager.default.temporaryDirectory.appendingPathComponent("ragdrop-child-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: marker) }
        let task = Task {
            try await ProcessRunner.run(executable: "/bin/sh", arguments: ["-c", "sleep 30 & echo $! > \"$1\"; wait", "fixture", marker.path], label: "descendant", timeout: 10)
        }
        for _ in 0..<40 {
            if FileManager.default.fileExists(atPath: marker.path) { break }
            try await Task.sleep(for: .milliseconds(50))
        }
        let pid = try #require(Int32(String(contentsOf: marker, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
        task.cancel()
        do { _ = try await task.value; Issue.record("Cancellation ignored") } catch is CancellationError {}
        // A zombie may briefly await reaping; it cannot continue writing.
        if kill(pid, 0) == 0 {
            let status = try await ProcessRunner.run(executable: "/bin/ps", arguments: ["-p", String(pid), "-o", "stat="], label: "descendant check", timeout: 5)
            #expect(status.output.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("Z"))
        }
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAGDROP_INSTALL_TEST_ROOT"] != nil))
    func cleanInstallAndRepairWithNoSystemPython() async throws {
        let support = URL(fileURLWithPath: ProcessInfo.processInfo.environment["RAGDROP_INSTALL_TEST_ROOT"]!)
        let resources = URL(fileURLWithPath: ProcessInfo.processInfo.environment["RAGDROP_INSTALL_TEST_RESOURCES"]!)
        let engine = LocalEngine(support: support, resources: resources)
        let library = support.appendingPathComponent("Papers with spaces – été")
        let connection = try await engine.prepare(library: library) { print($0) }
        #expect(URL(fileURLWithPath: connection.python).resolvingSymlinksInPath().path.hasPrefix(support.appendingPathComponent("Python").path))
        let render = try await ProcessRunner.run(executable: connection.python, arguments: ["-c", "import pypdf,pypandoc; print(pypandoc.convert_text('# Test\\n\\n| A | B |\\n|---|---|\\n| 1 | 2 |', 'html5', format='gfm'))"], label: "packaged renderer", timeout: 30)
        #expect(render.output.contains("<table>") && render.output.contains("<h1"))
        let status: LocalLibraryStatus = try await engine.decoded("status")
        #expect(status.ready && status.documents == 0 && status.chunks == 0)
        let articles: [HistoryDocument] = try await engine.decoded("documents")
        #expect(articles.isEmpty)
        let sentinel = library.appendingPathComponent("keep.txt")
        try "preserve".write(to: sentinel, atomically: true, encoding: .utf8)
        _ = try await engine.prepare(library: library) { print($0) }
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "preserve")
    }
}
