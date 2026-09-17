import SwiftUI

struct BatchProgressView: View {
    let jobs: [ImportJob]
    let message: String

    private let pipeline: [(ImportStage, String, String)] = [
        (.converting, "Mistral", "doc.text.magnifyingglass"),
        (.transferring, "NAS", "arrow.up.circle"),
        (.indexing, "Indexation", "square.stack.3d.up"),
        (.verifying, "Vérification", "checkmark.shield")
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Progression du lot")
                        .font(.headline)
                    Text(batchStateLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text("\(Int(overallProgress * 100)) %")
                    .font(.title3.monospacedDigit().weight(.semibold))
            }

            ProgressView(value: overallProgress)
                .progressViewStyle(.linear)

            HStack(spacing: 6) {
                ForEach(Array(pipeline.enumerated()), id: \.offset) { index, step in
                    stepView(stage: step.0, title: step.1, symbol: step.2)
                    if index < pipeline.count - 1 {
                        Rectangle()
                            .fill(connectorColor(after: step.0))
                            .frame(height: 2)
                    }
                }
            }

            if let activeJob {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    HStack(spacing: 6) {
                        ProgressView()
                            .controlSize(.small)
                        Text("PDF \(activeOrdinal) sur \(jobs.count) · \(activeJob.displayName)")
                            .lineLimit(1)
                        Text("• \(elapsed(from: activeJob.stageStartedAt, to: context.date))")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    .font(.caption)
                }
            } else {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .background(.regularMaterial, in: .rect(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(.separator.opacity(0.45))
        }
    }

    private func stepView(stage: ImportStage, title: String, symbol: String) -> some View {
        let state = stepState(for: stage)
        return VStack(spacing: 5) {
            Group {
                if state == .active {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: state == .complete ? "checkmark.circle.fill" : symbol)
                        .foregroundStyle(state == .complete ? Color.green : Color.secondary)
                }
            }
            .frame(height: 18)
            Text(title)
                .font(.caption2)
                .foregroundStyle(state == .active ? Color.primary : Color.secondary)
        }
        .frame(minWidth: 66)
    }

    private enum StepState { case pending, active, complete }

    private func stepState(for stage: ImportStage) -> StepState {
        guard let focusJob else { return .pending }
        if focusJob.stage == .completed { return .complete }
        guard let current = pipeline.firstIndex(where: { $0.0 == focusJob.stage }),
              let target = pipeline.firstIndex(where: { $0.0 == stage }) else {
            if focusJob.stage == .readyForIndexing || focusJob.stage == .awaitingReview,
               let targetProgress = stage.progressFraction,
               targetProgress < focusJob.overallProgress {
                return .complete
            }
            return .pending
        }
        if target < current { return .complete }
        if target == current { return .active }
        return .pending
    }

    private func connectorColor(after stage: ImportStage) -> Color {
        stepState(for: stage) == .complete ? .green.opacity(0.7) : .secondary.opacity(0.2)
    }

    private var activeJob: ImportJob? {
        jobs.first(where: { $0.stage.isActive })
    }

    private var focusJob: ImportJob? {
        activeJob ?? jobs.first(where: { $0.stage == .failed || $0.stage == .queued }) ?? jobs.last
    }

    private var overallProgress: Double {
        guard !jobs.isEmpty else { return 0 }
        return jobs.map(phaseProgress).reduce(0, +) / Double(jobs.count)
    }

    private var activeOrdinal: Int {
        guard let activeJob, let index = jobs.firstIndex(where: { $0.id == activeJob.id }) else { return 1 }
        return index + 1
    }

    private func phaseProgress(_ job: ImportJob) -> Double {
        job.overallProgress
    }

    private var progressLabel: String {
        let completed = jobs.filter { $0.stage == .completed }.count
        let failed = jobs.filter { $0.stage == .failed }.count
        if failed > 0 { return "\(completed) réussis, \(failed) en échec, sur \(jobs.count)" }
        return "\(completed) sur \(jobs.count) PDF ajoutés"
    }

    private var batchStateLabel: String {
        if let activeJob {
            return "PDF \(activeOrdinal) sur \(jobs.count) · \(activeJob.stage.title)"
        }
        let waiting = jobs.filter { $0.stage == .queued }.count
        let review = jobs.filter { $0.stage == .awaitingReview }.count
        let approved = jobs.filter { $0.stage == .readyForIndexing }.count
        if review > 0 { return "\(review) Markdown à vérifier avant l’envoi" }
        let indexingRetries = jobs.filter {
            $0.stage == .readyForIndexing && $0.detail.hasPrefix("Indexation à reprendre")
        }.count
        if indexingRetries > 0 {
            return "Indexation à reprendre pour \(indexingRetries) PDF"
        }
        if approved > 0 { return "\(approved) PDF approuvés, prêts pour Ragdoc" }
        if waiting > 0 { return "\(waiting) PDF prêts · lancez l’analyse du lot" }
        return progressLabel
    }

    private func elapsed(from start: Date, to end: Date) -> String {
        let total = max(0, Int(end.timeIntervalSince(start)))
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}
