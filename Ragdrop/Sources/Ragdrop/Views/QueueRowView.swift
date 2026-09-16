import SwiftUI

struct QueueRowView: View {
    let job: ImportJob
    var onPreview: (() -> Void)?

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: job.stage.symbol)
                .font(.title3)
                .foregroundStyle(stageColor)
                .frame(width: 24)
                .symbolEffect(.pulse, isActive: job.stage.isActive)

            VStack(alignment: .leading, spacing: 3) {
                Text(job.displayName)
                    .lineLimit(1)
                Text(job.detail)
                    .font(.caption)
                    .foregroundStyle(job.stage == .failed ? Color.red : Color.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 12)
            if job.stage == .awaitingReview || job.stage == .readyForIndexing {
                Button("Aperçu", systemImage: "eye", action: { onPreview?() })
                    .labelStyle(.iconOnly)
                    .help("Prévisualiser le Markdown")
            }
            Text(job.stage.title)
                .font(.caption.weight(.medium))
                .foregroundStyle(stageColor)
        }
        .padding(.vertical, 5)
    }

    private var stageColor: Color {
        switch job.stage {
        case .completed: .green
        case .awaitingReview: .orange
        case .readyForIndexing: .green
        case .duplicate: .orange
        case .rejected: .secondary
        case .failed: .red
        case .queued: .secondary
        default: .accentColor
        }
    }
}
