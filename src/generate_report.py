"""Generate the final PDF report from notebook-exported result tables."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "phishing_url_detection_report.pdf"
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2B6F9F")
PALE_BLUE = colors.HexColor("#EAF2F8")
ORANGE = colors.HexColor("#D97732")
GREY = colors.HexColor("#5B6570")
LIGHT_GREY = colors.HexColor("#F4F6F7")


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "cv": REPORTS / "cross_validation_results.csv",
        "threshold": REPORTS / "threshold_tuning.csv",
        "final": REPORTS / "final_test_results.csv",
        "importance": REPORTS / "feature_importance.csv",
    }
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run the notebook first; missing result files: {missing}")
    return tuple(pd.read_csv(paths[name]) for name in ("cv", "threshold", "final", "importance"))


def make_charts(temp: Path, cv: pd.DataFrame, threshold: pd.DataFrame,
                final: pd.DataFrame, importance: pd.DataFrame) -> dict[str, Path]:
    sns.set_theme(style="whitegrid")
    chart_paths: dict[str, Path] = {}

    f1 = cv.loc[cv["metric"].eq("f1")].copy()
    path = temp / "cv_f1.png"
    errors = np.vstack([f1["mean"] - f1["ci95_low"], f1["ci95_high"] - f1["mean"]])
    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.errorbar(f1["model"], f1["mean"], yerr=errors, fmt="o", capsize=6,
                  color="#2B6F9F", markersize=8)
    axis.set_ylim(0.86, 0.98)
    axis.set_ylabel("Phishing-class F1")
    axis.set_title("Development group cross-validation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_paths["cv"] = path

    path = temp / "threshold.png"
    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.plot(threshold["threshold"], threshold["phishing_precision"], marker="o",
              label="Precision")
    axis.plot(threshold["threshold"], threshold["phishing_recall"], marker="o",
              label="Recall")
    selected = float(final.iloc[0]["threshold"])
    axis.axvline(selected, color="#D97732", linestyle="--", label=f"Selected = {selected:.2f}")
    axis.set_ylim(0.84, 1.01)
    axis.set_xlabel("Phishing probability threshold")
    axis.set_ylabel("Score")
    axis.set_title("Out-of-fold development threshold trade-off")
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_paths["threshold"] = path

    row = final.iloc[0]
    matrix = np.array([
        [int(row["true_negative"]), int(row["legitimate_blocked_fp"])],
        [int(row["missed_phishing_fn"]), int(row["true_positive"])],
    ])
    path = temp / "confusion.png"
    fig, axis = plt.subplots(figsize=(5.0, 4.0))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Final untouched test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_paths["confusion"] = path

    top = importance.head(10).sort_values("importance")
    path = temp / "importance.png"
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.barh(top["feature"], top["importance"], color="#4C8C6B")
    axis.set_xlabel("Impurity importance")
    axis.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_paths["importance"] = path
    return chart_paths


def page_number(canvas, doc) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#D7DEE5"))
    canvas.line(0.65 * inch, 0.55 * inch, width - 0.65 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.7 * inch, 0.35 * inch, "Reproducible Phishing URL Detection Evaluation")
    canvas.drawRightString(width - 0.7 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def make_table(data, widths=None, header=True, font_size=8.2) -> KeepTogether:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#24313C")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D2DA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
        for row in range(1, len(data)):
            if row % 2 == 0:
                commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT_GREY))
    table.setStyle(TableStyle(commands))
    return KeepTogether([table])


def build_report() -> Path:
    cv, threshold, final, importance = load_results()
    result = final.iloc[0]
    cv_f1 = cv.loc[cv["metric"].eq("f1")].sort_values("mean", ascending=False)
    selected_threshold = threshold.iloc[(threshold["threshold"] - result["threshold"]).abs().argmin()]

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=24, leading=29, textColor=NAVY, alignment=TA_LEFT, spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", parent=styles["Normal"], fontSize=12.5, leading=18,
        textColor=GREY, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=15, leading=18, textColor=NAVY, spaceBefore=12, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=9, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BodyReport", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.4, leading=13.4, textColor=colors.HexColor("#27333D"), spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=9.5, leading=13.5, textColor=NAVY, backColor=PALE_BLUE,
        borderColor=BLUE, borderWidth=0.7, borderPadding=9, spaceBefore=6, spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique",
        fontSize=8, leading=10, textColor=GREY, alignment=TA_CENTER, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["BodyText"], fontSize=8.2, leading=11.2,
        textColor=GREY, spaceAfter=5,
    ))

    body = styles["BodyReport"]
    section = styles["Section"]
    subsection = styles["Subsection"]
    story = []

    with TemporaryDirectory() as temp_dir:
        charts = make_charts(Path(temp_dir), cv, threshold, final, importance)

        story.extend([
            Spacer(1, 0.75 * inch),
            Paragraph("Critical Evaluation of Phishing URL Detection", styles["ReportTitle"]),
            Paragraph("A leakage-aware, reproducible machine-learning benchmark", styles["Subtitle"]),
            Spacer(1, 0.28 * inch),
            Table([["PROJECT", "Data Science in Cybersecurity"],
                   ["SUBMISSION", "raghdans/phishing_url_detection_submission2"],
                   ["EVALUATED SOURCE", "vaibhavbichave/Phishing-URL-Detection"],
                   ["REPRODUCIBILITY SEED", "42"],
                   ["REPORT DATE", "4 July 2026"]],
                  colWidths=[1.7 * inch, 4.7 * inch],
                  style=TableStyle([
                      ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                      ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                      ("FONTSIZE", (0, 0), (-1, -1), 9),
                      ("TEXTCOLOR", (0, 0), (0, -1), BLUE),
                      ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#27333D")),
                      ("LINEBELOW", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7DEE5")),
                      ("TOPPADDING", (0, 0), (-1, -1), 8),
                      ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                  ])),
            Spacer(1, 0.45 * inch),
            Paragraph(
                "Central finding", styles["Subsection"]),
            Paragraph(
                "Random Forest remains the strongest tested model after strict leakage control, "
                f"achieving F1 {result['f1']:.4f} and ROC-AUC {result['roc_auc']:.4f} on an untouched, "
                "group-separated final test set. The lower result relative to the earlier random-split "
                "estimate is evidence of a more credible evaluation, not a weaker project.",
                styles["Callout"],
            ),
            Spacer(1, 1.15 * inch),
            Paragraph("Prepared as a reproducible academic evaluation. Results describe this dataset and do not establish production readiness.", styles["Small"]),
            PageBreak(),
        ])

        story.extend([
            Paragraph("Executive Summary", section),
            Paragraph(
                "This study reproduces and extends a public phishing URL classification project. "
                "It compares Logistic Regression, Random Forest, and Gradient Boosting using 30 "
                "pre-engineered numerical indicators. Phishing is encoded as the positive class so "
                "recall and false negatives have direct security meaning.", body),
            Paragraph(
                "The main methodological contribution is control of exact-feature duplication. The "
                "11,054-row dataset contains only 5,785 unique predictor vectors; 5,269 rows repeat a "
                "vector seen elsewhere, and 64 vector groups contain conflicting labels. A conventional "
                "random split can place identical vectors in training and testing, creating information "
                "leakage. This evaluation assigns every identical vector to one group and keeps each "
                "group within a single partition and cross-validation fold.", body),
            Paragraph(
                "Model selection uses five-fold stratified group cross-validation on development data. "
                "The Random Forest is selected by mean phishing-class F1. Its decision threshold is then "
                "selected from out-of-fold development probabilities, maximizing phishing recall while "
                "requiring at least 95% phishing precision. Only after those choices are fixed is the "
                "untouched final test partition evaluated.", body),
            make_table([
                ["Final model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "MCC", "ROC-AUC"],
                [result["model"], f"{result['threshold']:.2f}", f"{result['accuracy']:.4f}",
                 f"{result['phishing_precision']:.4f}", f"{result['phishing_recall']:.4f}",
                 f"{result['f1']:.4f}", f"{result['mcc']:.4f}", f"{result['roc_auc']:.4f}"],
            ], widths=[1.2*inch, .62*inch, .67*inch, .72*inch, .62*inch, .58*inch, .58*inch, .68*inch], font_size=7.5),
            Spacer(1, 6),
            Paragraph(
                f"The final test contains {int(result['missed_phishing_fn'])} missed phishing samples "
                f"and {int(result['legitimate_blocked_fp'])} blocked legitimate samples. Strong ranking "
                "performance does not remove the operational cost of either error type.", styles["Callout"]),
            Paragraph("1. Research Question and Scope", section),
            Paragraph(
                "The research question is: How reliably do a linear baseline and nonlinear ensembles "
                "distinguish phishing from legitimate websites on the supplied feature dataset when "
                "model selection, threshold selection, and duplicate leakage are explicitly controlled?", body),
            Paragraph(
                "The scope is deliberately narrower than real-time URL detection. The CSV contains "
                "already-computed indicators rather than raw URLs. Consequently, this study evaluates "
                "classification under the dataset's representation; it does not validate feature "
                "collection latency, live domain lookup reliability, or resistance to adaptive attackers.", body),
        ])

        story.extend([
            Paragraph("2. Source and Dataset Evaluation", section),
            Paragraph("2.1 Selected source", subsection),
            Paragraph(
                "The selected source is the public GitHub repository Phishing-URL-Detection by Vaibhav "
                "Bichave. It provides code and a CSV, making basic reproduction possible. Its documentation "
                "does not, however, fully establish URL collection dates, sampling frames, label adjudication, "
                "or whether related observations were grouped during evaluation. Those omissions limit "
                "external validity and make high headline accuracy insufficient evidence for deployment.", body),
            Paragraph("2.2 Data-quality audit", subsection),
            make_table([
                ["Audit item", "Result", "Interpretation"],
                ["Rows", "11,054", "Sample-level observations"],
                ["Model predictors", "30", "Identifier and target excluded"],
                ["Missing values", "0", "No imputation activated on this snapshot"],
                ["Unique feature vectors", "5,785", "Substantial exact repetition"],
                ["Rows beyond first group member", "5,269", "Potential random-split leakage"],
                ["Conflicting-label groups", "64", "Identical predictors, inconsistent targets"],
            ], widths=[1.75*inch, 1.15*inch, 3.55*inch]),
            Spacer(1, 8),
            Paragraph(
                "The identifier column previously masked the duplication because every row identifier is "
                "unique. Auditing duplicates only across all raw columns therefore returns zero and is "
                "misleading for modeling. The relevant audit is performed after removing the identifier "
                "and, separately, after removing both identifier and target.", body),
            Paragraph(
                "Conflicting-label groups imply irreducible ambiguity under the available representation: "
                "the model receives the same predictors for samples assigned different labels. This may "
                "reflect label noise, coarse feature discretization, or unobserved information. The study "
                "retains these rows but prevents their groups from crossing folds.", styles["Callout"]),
            Paragraph("2.3 Label semantics", subsection),
            Paragraph(
                "The source labels phishing as -1 and legitimate as 1. This reproduction maps phishing to "
                "1 and legitimate to 0. Therefore phishing recall measures the proportion of phishing "
                "samples detected, and a false negative is a phishing sample incorrectly allowed as "
                "legitimate. This convention removes the ambiguity present in the earlier report.", body),
            PageBreak(),
        ])

        story.extend([
            Paragraph("3. Experimental Design", section),
            Paragraph("3.1 Leakage-aware partitions", subsection),
            Paragraph(
                "Exact predictor vectors are factorized into groups. StratifiedGroupKFold with five folds, "
                "shuffle enabled, and random state 42 supplies candidate outer partitions. The candidate "
                "closest to the desired 20% size and global phishing rate becomes the final test partition. "
                "This yields 8,827 development rows and 2,227 final-test rows, with zero shared groups.", body),
            make_table([
                ["Partition", "Rows", "Phishing rate", "Used for"],
                ["Development", "8,827", "0.4417", "CV model selection and OOF threshold selection"],
                ["Final test", "2,227", "0.4481", "One evaluation after all choices are fixed"],
            ], widths=[1.25*inch, .7*inch, 1.0*inch, 3.55*inch]),
            Paragraph("3.2 Models and preprocessing", subsection),
            Paragraph(
                "Logistic Regression provides an interpretable linear baseline and uses median imputation "
                "plus standardization inside its pipeline. Random Forest uses 300 trees, class balancing, "
                "and a fixed seed. Gradient Boosting supplies a second nonlinear comparator. Tree pipelines "
                "include median imputation but no scaling. Fitting preprocessing inside each fold prevents "
                "validation information from affecting training transformations.", body),
            Paragraph("3.3 Metrics and uncertainty", subsection),
            Paragraph(
                "Accuracy is reported for comparability, but the primary selection metric is phishing-class "
                "F1. Precision measures how often phishing alerts are correct; recall measures how much "
                "phishing is detected. MCC uses all four confusion-matrix cells and is informative under "
                "class imbalance. ROC-AUC evaluates ranking across thresholds. Fold means, sample standard "
                "deviations, and normal-approximation 95% intervals summarize split variability. With only "
                "five correlated training folds, these intervals are descriptive rather than population guarantees.", body),
            PageBreak(),
            Paragraph("4. Development Results", section),
            Image(str(charts["cv"]), width=6.55*inch, height=3.27*inch),
            Paragraph("Figure 1. Mean phishing-class F1 with approximate 95% intervals across grouped folds.", styles["Caption"]),
        ])

        cv_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "MCC", "ROC-AUC"]]
        for model in cv_f1["model"]:
            subset = cv.loc[cv["model"].eq(model)].set_index("metric")
            cv_rows.append([
                model,
                *[f"{subset.loc[metric, 'mean']:.4f}" for metric in
                  ("accuracy", "phishing_precision", "phishing_recall", "f1", "mcc", "roc_auc")],
            ])
        story.extend([
            make_table(cv_rows, widths=[1.45*inch, .72*inch, .76*inch, .68*inch, .62*inch, .62*inch, .72*inch], font_size=7.8),
            Spacer(1, 7),
            Paragraph(
                f"Random Forest has the highest mean F1 ({cv_f1.iloc[0]['mean']:.4f}), followed by "
                f"Gradient Boosting ({cv_f1.iloc[1]['mean']:.4f}) and Logistic Regression "
                f"({cv_f1.iloc[2]['mean']:.4f}). The intervals overlap between the two tree ensembles, "
                "so the ranking should not be overstated as universal superiority. The selected model is "
                "simply the best under the predeclared development criterion.", body),
            KeepTogether([
                Paragraph("5. Threshold Selection", section),
                Image(str(charts["threshold"]), width=6.55*inch, height=3.27*inch),
                Paragraph("Figure 2. Threshold choice uses out-of-fold development predictions only.", styles["Caption"]),
            ]),
            Paragraph(
                f"The selected threshold is {result['threshold']:.2f}. On out-of-fold development "
                f"predictions it produces precision {selected_threshold['phishing_precision']:.4f}, recall "
                f"{selected_threshold['phishing_recall']:.4f}, and F1 {selected_threshold['f1']:.4f}. It is "
                "the threshold with maximum recall among candidates meeting the 0.95 precision floor. The "
                "precision floor is a transparent study assumption, not a claim about every organization's costs.", body),
            PageBreak(),
        ])

        story.extend([
            Paragraph("6. Final Test Evaluation", section),
            make_table([
                ["Metric", "Value", "Security interpretation"],
                ["Accuracy", f"{result['accuracy']:.4f}", "Overall proportion correct"],
                ["Phishing precision", f"{result['phishing_precision']:.4f}", "Reliability of phishing alerts"],
                ["Phishing recall", f"{result['phishing_recall']:.4f}", "Fraction of phishing detected"],
                ["Phishing F1", f"{result['f1']:.4f}", "Balance of phishing precision and recall"],
                ["MCC", f"{result['mcc']:.4f}", "Balanced correlation across all outcomes"],
                ["ROC-AUC", f"{result['roc_auc']:.4f}", "Threshold-independent ranking quality"],
            ], widths=[1.45*inch, .85*inch, 4.2*inch]),
            Spacer(1, 8),
            Image(str(charts["confusion"]), width=4.6*inch, height=3.68*inch),
            Paragraph("Figure 3. Phishing is the positive class; the lower-left cell contains missed phishing.", styles["Caption"]),
            Paragraph(
                f"Of 998 phishing samples, {int(result['true_positive'])} are detected and "
                f"{int(result['missed_phishing_fn'])} are missed. Of 1,229 legitimate samples, "
                f"{int(result['true_negative'])} are allowed and {int(result['legitimate_blocked_fp'])} "
                "are blocked. The false negatives are the higher-risk security failures because they may "
                "expose users to credential theft or malware; false positives impose usability, support, "
                "and business-continuity costs.", body),
            Paragraph("6.1 Why the corrected score is lower", subsection),
            Paragraph(
                "The earlier submission reported final-test F1 near 0.971 under an ordinary random split. "
                "The leakage-aware evaluation reports 0.934. Because nearly half the rows repeat a feature "
                "vector, random splitting can let a model encounter the same vector during training and "
                "testing. Group separation removes this shortcut. The corrected result is therefore the "
                "appropriate headline figure for this submission.", styles["Callout"]),
        ])

        story.extend([
            Paragraph("7. Interpretation", section),
            Image(str(charts["importance"]), width=6.55*inch, height=3.82*inch),
            Paragraph("Figure 4. Impurity-based importance describes model usage, not causal effects.", styles["Caption"]),
            Paragraph(
                "HTTPS and AnchorURL dominate Random Forest importance, followed by WebsiteTraffic and "
                "SubDomains. This is consistent with the dataset's engineered representation, but it does "
                "not prove that changing one indicator causes a site to become phishing or legitimate. "
                "Impurity importance may also favor variables offering more candidate splits and can divide "
                "credit among correlated predictors. Permutation importance on future, grouped data would "
                "provide a useful robustness check.", body),
            Paragraph("8. Threats to Validity", section),
            Paragraph("Internal validity", subsection),
            Paragraph(
                "Exact-match grouping addresses a major leakage route but cannot detect near-duplicates, "
                "shared domains, repeated campaigns, or preprocessing performed before the dataset was "
                "published. Conflicting labels also cap achievable consistency. Hyperparameters are largely "
                "fixed rather than searched, which limits optimization but reduces selection overfitting.", body),
            Paragraph("External validity", subsection),
            Paragraph(
                "No timestamps support train-past/test-future evaluation. Phishing tactics, hosting, browser "
                "behavior, and reputation services drift. Random grouped folds estimate performance within "
                "this historical dataset, not on future campaigns or different organizations.", body),
            Paragraph("Construct and operational validity", subsection),
            Paragraph(
                "The dataset measures discretized indicators rather than raw URL text or a complete browsing "
                "context. Some features may require live network or third-party reputation queries. Their "
                "availability, latency, failure modes, privacy implications, and susceptibility to attacker "
                "manipulation are outside the experiment.", body),
            Paragraph("Statistical conclusion validity", subsection),
            Paragraph(
                "Five folds provide a limited view of variability, and fold estimates share training data. "
                "The normal-approximation intervals are therefore descriptive. A stronger study would use "
                "repeated grouped splits or a group-level bootstrap.", body),
        ])

        story.extend([
            Paragraph("9. Reproducibility", section),
            Paragraph(
                "The repository now matches its documentation. The dataset is under data/, the executed "
                "notebook is under notebooks/, reusable logic is under src/, report artifacts are under "
                "reports/, and tests are under tests/. Dependencies are version-pinned. Paths are resolved "
                "from the repository rather than from /content or a personal directory. Randomness is fixed, "
                "schema and label assumptions are validated, and four automated tests cover loading, label "
                "semantics, deterministic splitting, and group separation.", body),
            Paragraph(
                "A reviewer can create a virtual environment, install requirements.txt, execute the notebook "
                "with nbconvert, run pytest, and regenerate this report with python -m src.generate_report. "
                "The committed notebook is already executed, while the CSV tables provide compact, machine-"
                "readable evidence for every headline result.", body),
            Paragraph("10. Conclusions", section),
            Paragraph(
                "Random Forest is the strongest of the three tested models under grouped development "
                "cross-validation and achieves strong discrimination on the untouched final test. More "
                "important than the model ranking, however, is the correction of evaluation leakage. The "
                "project demonstrates that repository organization, label semantics, data-quality auditing, "
                "and partition design can materially change the credibility of a cybersecurity result.", body),
            Paragraph(
                "The study supports a careful claim: engineered website indicators can classify this dataset "
                "with high performance when exact duplicate vectors are separated. It does not support a "
                "claim that the model is deployment-ready. Production evidence would require fresh temporally "
                "separated URLs, domain- or campaign-level grouping, calibrated probabilities, cost-based "
                "thresholds, adversarial evaluation, and continuous drift monitoring.", styles["Callout"]),
            Paragraph("11. Future Work", section),
            Paragraph(
                "Priority extensions are: (1) obtain timestamped raw URLs with documented sampling and label "
                "adjudication; (2) group by registered domain and campaign rather than only exact vectors; "
                "(3) quantify near-duplicate similarity; (4) evaluate calibration and decision curves; "
                "(5) use nested grouped validation for any hyperparameter search; (6) test on an external, "
                "newer dataset; and (7) measure feature extraction latency, failure, privacy, and drift.", body),
            Paragraph("References", section),
            Paragraph(
                "[1] V. Bichave, Phishing-URL-Detection, GitHub repository, "
                "https://github.com/vaibhavbichave/Phishing-URL-Detection.", body),
            Paragraph(
                "[2] D. M. W. Powers, Evaluation: From Precision, Recall and F-Measure to ROC, "
                "Informedness, Markedness and Correlation, Journal of Machine Learning Technologies, 2011.", body),
            Paragraph(
                "[3] D. Chicco and G. Jurman, The advantages of the Matthews correlation coefficient over "
                "F1 score and accuracy in binary classification evaluation, BMC Genomics, 2020. "
                "doi:10.1186/s12864-019-6413-7.", body),
            Paragraph(
                "[4] F. Pedregosa et al., Scikit-learn: Machine Learning in Python, Journal of Machine "
                "Learning Research, 2011.", body),
        ])

        document = SimpleDocTemplate(
            str(OUTPUT), pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch,
            topMargin=0.62*inch, bottomMargin=0.72*inch,
            title="Critical Evaluation of Phishing URL Detection",
            author="Reproducible Data Science in Cybersecurity Project",
            subject="Leakage-aware phishing URL classification evaluation",
        )
        document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
