import Foundation
import Darwin

struct ProcessResult: Sendable {
    let output: String
    let error: String
}

private final class ProcessCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func cancel() { lock.lock(); value = true; lock.unlock() }
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return value }
}

struct ProcessRunner {
    static func run(executable: String, arguments: [String], standardInput: URL? = nil,
                    label: String, timeout: TimeInterval? = nil,
                    environment suppliedEnvironment: [String: String]? = nil) async throws -> ProcessResult {
        let cancellation = ProcessCancellation()
        return try await withTaskCancellationHandler {
            try await Task.detached(priority: .userInitiated) {
                let fm = FileManager.default
                let temp = fm.temporaryDirectory.appendingPathComponent("ragdrop-process-\(UUID().uuidString)")
                try fm.createDirectory(at: temp, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
                defer { try? fm.removeItem(at: temp) }
                let out = temp.appendingPathComponent("stdout")
                let err = temp.appendingPathComponent("stderr")
                fm.createFile(atPath: out.path, contents: nil, attributes: [.posixPermissions: 0o600])
                fm.createFile(atPath: err.path, contents: nil, attributes: [.posixPermissions: 0o600])
                let output = try FileHandle(forWritingTo: out)
                let error = try FileHandle(forWritingTo: err)
                let input = try FileHandle(forReadingFrom: standardInput ?? URL(fileURLWithPath: "/dev/null"))
                defer { try? output.close(); try? error.close(); try? input.close() }
                var env = suppliedEnvironment ?? ProcessInfo.processInfo.environment
                env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
                if cancellation.isCancelled { throw CancellationError() }
                let pid = try spawn(executable: executable, arguments: arguments, environment: env,
                                    input: input.fileDescriptor, output: output.fileDescriptor, error: error.fileDescriptor)
                var status: Int32 = 0
                var reaped = false
                var timedOut = false
                let deadline = timeout.map { Date().addingTimeInterval($0) }
                while !reaped {
                    let result = waitpid(pid, &status, WNOHANG)
                    if result == pid { reaped = true; break }
                    if result == -1, errno != EINTR { break }
                    if cancellation.isCancelled || (deadline.map { Date() >= $0 } ?? false) {
                        timedOut = !cancellation.isCancelled
                        kill(-pid, SIGTERM)
                        let grace = Date().addingTimeInterval(3)
                        while Date() < grace {
                            if !reaped, waitpid(pid, &status, WNOHANG) == pid { reaped = true }
                            if kill(-pid, 0) == -1 && errno == ESRCH { break }
                            try? await Task.sleep(for: .milliseconds(50))
                        }
                        break
                    }
                    try? await Task.sleep(for: .milliseconds(50))
                }
                // Descendants share the isolated group, including installers spawned by uv.
                // Also clean up a background child left behind by an exited command.
                kill(-pid, SIGKILL)
                if !reaped { while waitpid(pid, &status, 0) == -1 && errno == EINTR {} }
                try output.synchronize(); try error.synchronize()
                let stdout = String(decoding: try Data(contentsOf: out), as: UTF8.self)
                let stderr = String(decoding: try Data(contentsOf: err), as: UTF8.self)
                if cancellation.isCancelled { throw CancellationError() }
                if timedOut, let timeout { throw PipelineError.timedOut(command: label, minutes: max(1, Int(timeout / 60))) }
                guard status == 0 else {
                    throw PipelineError.processFailed(command: label, details: String((stderr.isEmpty ? stdout : stderr).suffix(8_000)))
                }
                return ProcessResult(output: stdout, error: stderr)
            }.value
        } onCancel: { cancellation.cancel() }
    }

    private static func spawn(executable: String, arguments: [String], environment: [String: String],
                              input: Int32, output: Int32, error: Int32) throws -> pid_t {
        var actions: posix_spawn_file_actions_t?
        var attributes: posix_spawnattr_t?
        posix_spawn_file_actions_init(&actions)
        posix_spawnattr_init(&attributes)
        defer { posix_spawn_file_actions_destroy(&actions); posix_spawnattr_destroy(&attributes) }
        posix_spawn_file_actions_adddup2(&actions, input, STDIN_FILENO)
        posix_spawn_file_actions_adddup2(&actions, output, STDOUT_FILENO)
        posix_spawn_file_actions_adddup2(&actions, error, STDERR_FILENO)
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT))
        posix_spawnattr_setpgroup(&attributes, 0)
        let argv = ([executable] + arguments).map { strdup($0) } + [nil]
        let envp = environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { argv.forEach { free($0) }; envp.forEach { free($0) } }
        var pid: pid_t = 0
        let result = argv.withUnsafeBufferPointer { args in
            envp.withUnsafeBufferPointer { env in
                posix_spawn(&pid, executable, &actions, &attributes, args.baseAddress!, env.baseAddress!)
            }
        }
        guard result == 0 else { throw POSIXError(POSIXErrorCode(rawValue: result) ?? .EIO) }
        return pid
    }
}
