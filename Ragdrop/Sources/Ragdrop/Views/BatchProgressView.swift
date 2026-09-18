import SwiftUI

struct BatchProgressView: View {
    let jobs: [ImportJob]
    let message: String
    var onReview: ((ImportJob) -> Void)?
    var onError: ((ImportJob) -> Void)?
    @State private var showingDetails = false
    private var presentation: BatchPresentation { BatchPresentation(jobs: jobs) }
    private var summary: BatchSummary { presentation.summary }

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            HStack(alignment: .top, spacing: 24) {
                VStack(alignment: .leading, spacing: 9) {
                    Text(presentation.headline).font(.system(size: 20, weight: .semibold)).tracking(-0.3)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(presentation.subtitle).font(.system(size: 14)).foregroundStyle(RagdropTheme.secondary)
                    if let context = presentation.context {
                        Label(context, systemImage: presentation.hasFailure ? "exclamationmark.circle" : "eye")
                            .font(.callout.weight(.medium))
                            .foregroundStyle(presentation.hasFailure ? RagdropTheme.warning : RagdropTheme.link)
                    }
                }
                Spacer(minLength: 0)
                Text(RagdropText.addedCount(summary.added, total: jobs.count))
                    .font(.system(size: 15, weight: .medium)).monospacedDigit().fixedSize()
            }
            HStack(alignment: .top, spacing: 10) {
                ForEach(BatchPhase.allCases, id: \.rawValue) { phase in
                    VStack(alignment: .leading, spacing: 11) {
                        BatchStageTrack(active: presentation.phase == phase,
                                        passed: presentation.hasPassed(phase),
                                        working: presentation.isWorking && presentation.phase == phase)
                            .frame(height: 6)
                        Text(phase.title).font(.system(size: 14, weight: presentation.phase == phase ? .semibold : .regular))
                            .foregroundStyle(presentation.phase == phase ? RagdropTheme.link : RagdropTheme.secondary)
                            .lineLimit(1).minimumScaleFactor(0.8)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel("\(phase.title) : \(stageDescription(phase))")
                }
            }.accessibilityElement(children: .contain)
            Rectangle().fill(RagdropTheme.line).frame(height: 1)
            if let job = presentation.focus {
                HStack(spacing: 16) {
                    Image(systemName: "doc.text").font(.system(size: 24, weight: .light))
                        .foregroundStyle(RagdropTheme.text).frame(width: 32, height: 42)
                    VStack(alignment: .leading, spacing: 7) {
                        Text(job.metadata?.title.nonBlank ?? job.displayName).font(.system(size: 15, weight: .semibold))
                            .lineLimit(2).help(job.metadata?.title ?? job.displayName)
                        if let metadata = articleMetadata(job) {
                            Text(metadata).font(.system(size: 14)).foregroundStyle(RagdropTheme.secondary).lineLimit(1)
                        }
                    }
                    Spacer(minLength: 0)
                    if [.awaitingReview, .readyForIndexing].contains(job.stage), let onReview {
                        Button("Review") { onReview(job) }.buttonStyle(RagdropSecondaryButtonStyle())
                    }
                    if job.errorDetails != nil, let onError {
                        Button("View error") { onError(job) }.foregroundStyle(RagdropTheme.warning)
                    }
                }
            }
            DisclosureGroup("Details", isExpanded: $showingDetails) {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Queued: \(summary.count(.queued)) · To review: \(summary.review) · Approved: \(summary.count(.readyForIndexing)) · Duplicates: \(summary.count(.duplicate)) · Rejected: \(summary.count(.rejected))")
                    Text(message)
                    if let job = presentation.focus {
                        Text(job.displayDetail)
                        if let converter = job.converterName { Text("Conversion: \(converter)") }
                        if job.stage.isActive {
                            TimelineView(.periodic(from: .now, by: 1)) { context in
                                Text("\(max(0, Int(context.date.timeIntervalSince(job.stageStartedAt)))) s in this state · remaining time unknown")
                                    .monospacedDigit()
                            }
                        }
                    }
                    Text("Segments track the displayed article. The counter includes only verified additions to Ragdoc. Indexing does not validate the scientific accuracy of the extraction.")
                }.font(.callout).foregroundStyle(RagdropTheme.secondary).padding(.top, 10)
            }.font(.body).foregroundStyle(RagdropTheme.secondary)
        }
        .padding(28)
        .ragdropPanel()
    }

    private func articleMetadata(_ job: ImportJob) -> String? {
        guard let metadata = job.metadata else { return nil }
        let authors = metadata.authors.joined(separator: ", ")
        let parts = [authors.nonBlank, metadata.year.map(String.init)].compactMap { $0 }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
    private func stageDescription(_ phase: BatchPhase) -> String {
        if presentation.phase == phase { return presentation.isWorking ? "in progress, duration unknown" : "waiting" }
        return presentation.hasPassed(phase) ? "passed" : "not marked complete"
    }
}

private struct BatchStageTrack: View {
    let active: Bool
    let passed: Bool
    let working: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var body: some View {
        GeometryReader { geometry in
            RoundedRectangle(cornerRadius: 2)
                .fill(active ? RagdropTheme.signal : (passed ? RagdropTheme.signal.opacity(0.42) : RagdropTheme.raised))
                .overlay {
                    if working && !reduceMotion {
                        TimelineView(.animation(minimumInterval: 1.0 / 24)) { context in
                            let travel = geometry.size.width + 80
                            let x = context.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.8) / 1.8 * travel - 80
                            LinearGradient(colors: [.clear, RagdropTheme.panel.opacity(0.65), .clear], startPoint: .leading, endPoint: .trailing)
                                .frame(width: 80).offset(x: x)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .clipShape(.rect(cornerRadius: 2))
                .overlay { RoundedRectangle(cornerRadius: 2).strokeBorder(active ? RagdropTheme.link : RagdropTheme.line) }
        }.accessibilityHidden(true)
    }
}
