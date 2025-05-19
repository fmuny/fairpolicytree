# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Replication Code for *Interpretable Policy Learning with Sensitive Attributes
# Nora Bearth, Michael Lechner, Jana Mareckova, Fabian Muny
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ Load packages and define parameters -----------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# Load packages
# install_github("fmuny/fairpolicytree")
library(fairpolicytree)
library(policytree)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(rcompanion)
library(BayesFactor)
library(patchwork)
library(stringr)
library(knitr)
library(kableExtra)
library(cluster)

# Data source path
DATA_PATH <- "Q:/SEW/Projekte/NFP77/BeLeMaMu.Fairness/Data_fair/cleaned"

# Method for ties in fairness adjustments
# One of "average", "first", "last", "random", "max", "min"
# see https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/rank
TIES_METHOD <- 'random'

# Depth of policy trees
PT_DEPTH <- 3
PT_SEARCH_DEPTH <- PT_DEPTH
# If PT_SEARCH_DEPTH = PT_DEPTH then an exact tree is built.
# If PT_SEARCH_DEPTH < PT_DEPTH then a hybrid tree is built.
# See https://grf-labs.github.io/policytree/reference/hybrid_policy_tree.html

# Set seed for replicability
SEED = 12345
set.seed(SEED)

# Save outputs
save <- TRUE
RESULTS_PATH <- paste0(
  "Q:/SEW/Projekte/NFP77/BeLeMaMu.Fairness/Results_fair/plots_tables/")
if(save){
  folder_name <- format(Sys.Date(), "%Y-%m-%d")
  RESULTS_PATH <- paste0(RESULTS_PATH, folder_name, "_depth", PT_DEPTH, "/")
  if (!dir.exists(RESULTS_PATH)) {
    dir.create(RESULTS_PATH)
  }
}

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ Load Empirical Data ---------------------------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# Load and prepare data (add original outcome): Training sample
outcomes_pr <- read.csv(file.path(DATA_PATH, "data_clean_pr.csv"))
training_df <- read.csv(file.path(DATA_PATH, "iates_pr.csv"))
outcomes_pr <- outcomes_pr %>% select(outcome) %>% slice(training_df$id_mcf+1)
training_df <- cbind(training_df, outcomes_pr)
# Evaluation sample
outcomes_ev <- read.csv(file.path(DATA_PATH, "data_clean_ev.csv"))
evaluation_df <- read.csv(file.path(DATA_PATH, "iates_ev.csv"))
outcomes_ev <- outcomes_ev %>% select(outcome) %>% slice(evaluation_df$id_mcf+1)
evaluation_df <- cbind(evaluation_df, outcomes_ev)

# Create combined sensitive attribute
training_df$sens_comb <- factor(apply(
  training_df[c("female", "swiss")], 1, function(x) paste(x, collapse = "_")))
evaluation_df$sens_comb <- factor(apply(
  evaluation_df[c("female", "swiss")], 1, function(x) paste(x, collapse = "_")))

# Load variable lists and mappings
columns <- read.csv(file.path(DATA_PATH, "columns.csv"))
mappings <- read.csv(file.path(DATA_PATH, "mappings.csv"))

# Define variables
VAR_S_NAME_ORD <- columns$S_ord[columns$S_ord > 0]
VAR_A_NAME_ORD <- columns$A_ord[columns$A_ord > 0]
VAR_ID_NAME <- "id_mcf"
VAR_D_NAME <- columns$D[columns$D > 0]
unique_D <- sort(unique(training_df[[VAR_D_NAME]]))
unique_D_ch <- as.character(sort(unique(training_df[[VAR_D_NAME]])))
nunique_D <- length(unique_D)
unique(training_df[[VAR_D_NAME]])
VAR_POLSCORE_NAME <- sapply(0:(nunique_D-1), function(x) paste0(
  "outcome_lc", x, "_un_lc_pot_eff"))

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ First descriptive analysis --------------------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# Plot variables of interest
df_plot <- training_df %>% 
  select(all_of(c(VAR_POLSCORE_NAME, VAR_A_NAME_ORD, VAR_S_NAME_ORD))) %>%
  pivot_longer(
    cols = all_of(c(VAR_POLSCORE_NAME, VAR_A_NAME_ORD, VAR_S_NAME_ORD)), 
    names_to = "variable", 
    values_to = "value")
print(ggplot(
    df_plot, aes(x = value)) +
    geom_histogram(aes(y = after_stat(density)), colour = 'gray', bins=32, fill = "white") +
    geom_density(alpha = 0.5) +
    theme_minimal() +
    facet_wrap(~variable, scales='free'))

# Compute frequencies of observed assignments and observed policy value
prog_freq_observed <- training_df %>%
  count(treatment6) %>%  
  mutate(proportion = n / sum(n))
for(i in c(cat("Program frequencies observed:\n"), print(prog_freq_observed))){i}


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ MQ-adjustment ---------------------------------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# Run mq adjustments
datasets <- list(training = training_df, evaluation = evaluation_df)
vars_list <- list(scores = VAR_POLSCORE_NAME, As = VAR_A_NAME_ORD)
scores_adjusted <- list()
for (data_name in names(datasets)) {
  df <- datasets[[data_name]]
  scores_adjusted[[data_name]] <- list()
  for (vars in names(vars_list)) {
    cols <- vars_list[[vars]]
    scores_adjusted[[data_name]][[vars]] <- mq_adjustment(
      vars = df[cols],
      sens = df[VAR_S_NAME_ORD],
      seed = SEED,
      ties.method = TIES_METHOD
    )
  }
}
# Extract results
scores_org <- training_df[VAR_POLSCORE_NAME]
scores_cdf <- scores_adjusted$training$scores$vars_cdf
scores_mq <- scores_adjusted$training$scores$vars_mq
As_org <- training_df[VAR_A_NAME_ORD]
As_cdf <- scores_adjusted$training$As$vars_cdf
As_mq <- scores_adjusted$training$As$vars_mq
sens_org <- training_df[VAR_S_NAME_ORD]
sens_comb_org <- training_df['sens_comb']

scores_org_out <- evaluation_df[VAR_POLSCORE_NAME]
scores_cdf_out <- scores_adjusted$evaluation$scores$vars_cdf
scores_mq_out <- scores_adjusted$evaluation$scores$vars_mq
As_org_out <- evaluation_df[VAR_A_NAME_ORD]
As_cdf_out <- scores_adjusted$evaluation$As$vars_cdf
As_mq_out <- scores_adjusted$evaluation$As$vars_mq
sens_org_out <- evaluation_df[VAR_S_NAME_ORD]
sens_comb_org_out <- evaluation_df['sens_comb']

# Put into one df
training_fair_df <- cbind(training_df, scores_cdf, scores_mq, As_cdf, As_mq)
evaluation_fair_df <- cbind(evaluation_df, scores_cdf_out, scores_mq_out, As_cdf_out, As_mq_out)

# Print means and standard deviations
for(i in c(
  cat("Means and standard deviations of policy scores before and after adjustment\n"),
  print(data.frame(
    scores_mean = colMeans(training_df[VAR_POLSCORE_NAME]),
    scores_cdf_mean = colMeans(scores_cdf),
    scores_mq_mean = colMeans(scores_mq),
    scores_std = sqrt(diag(var(training_df[VAR_POLSCORE_NAME]))),
    scores_cdf_std = sqrt(diag(var(scores_cdf))),
    scores_mq_std = sqrt(diag(var(scores_mq))))),
  cat("\nMeans and standard deviations of decision variables before and after adjustment\n"),
  print(data.frame(
    As_mean = colMeans(training_df[VAR_A_NAME_ORD]),
    As_cdf_mean = colMeans(As_cdf),
    As_mq_mean = colMeans(As_mq),
    As_std = sqrt(diag(var(training_df[VAR_A_NAME_ORD]))),
    As_cdf_std = sqrt(diag(var(As_cdf))),
    As_mq_std = sqrt(diag(var(As_mq))))))){
  i
}

# Print correlation matrix
df_corr <- cbind(
  scores_org, scores_cdf, scores_mq, As_org, As_cdf, As_mq, sens_org)
correlations_S = data.frame(cor(df_corr)[, VAR_S_NAME_ORD])
colnames(correlations_S) <- VAR_S_NAME_ORD
print(round(correlations_S, 7))

correlations_A = data.frame(cor(df_corr)[
  c(VAR_POLSCORE_NAME, VAR_S_NAME_ORD),
  c(VAR_A_NAME_ORD, paste0(VAR_A_NAME_ORD, "_cdf"), paste0(VAR_A_NAME_ORD, "_mq"))])
print(round(correlations_A, 7))


# Plot distributions of scores and A for combinations of S
vars_to_plot <- c(VAR_POLSCORE_NAME, VAR_A_NAME_ORD)
vars_to_plot <- c(vars_to_plot, paste0(vars_to_plot, "_mq"))
vars_to_plot_sens <- c(vars_to_plot, 'sens_comb')

map_prog <- setNames(as.character(mappings[!is.na(mappings[VAR_D_NAME]),'X']), as.character(mappings[!is.na(mappings[VAR_D_NAME]), VAR_D_NAME]))

df_fullplot <- training_fair_df[, vars_to_plot_sens] %>% pivot_longer(all_of(vars_to_plot))
df_fullplot <- df_fullplot %>% mutate(
  type = if_else(str_ends(name, "_mq"), "Adjusted variable", "Original variable"),
  name = str_remove(name, "_mq"),
  program_number = str_match(name, "outcome_lc([0-9]+)_un_lc_pot_eff")[, 2],
  name = if_else(!is.na(program_number), paste0("Policy score (", map_prog[program_number], ")"), name),
  name = if_else(name == "age", "Age", name),
  name = if_else(name == "past_income", "Past earnings", name),
  name = if_else(name == "qual_degree", "Degree", name)
)

df_fullplot <- df_fullplot %>% mutate(name_type = paste0(name, "\n(", type, ")"))
df_fullplot$type <- factor(df_fullplot$type, levels = c("Original variable", "Adjusted variable"))
levs <- c(
  sort(unique(df_fullplot$name_type[str_ends(df_fullplot$name_type, "\\(Original variable\\)")])),
  sort(unique(df_fullplot$name_type[str_ends(df_fullplot$name_type, "\\(Adjusted variable\\)")]))
)
df_fullplot$name_type <- factor(df_fullplot$name_type, levels = levs)

df_fullplot_main <- df_fullplot[is.na(df_fullplot$program_number) | df_fullplot$program_number == 0,]
df_fullplot_appendix <- df_fullplot[!is.na(df_fullplot$program_number) & df_fullplot$program_number > 0,]

histograms_cleaning <- ggplot(df_fullplot_main, aes(x = value, fill = sens_comb)) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.5, position = "identity", bins = 32) +
  labs(x=NULL, y = "Density", fill="Sensitive attribue:") +
  theme_minimal() +
  facet_wrap(~name_type, scales = "free", ncol = 4) +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  scale_fill_discrete(
    labels = c("0_0" = "Foreign men", "1_0" = "Foreign women", "0_1" = "Swiss men", "1_1" = "Swiss women")) +
  theme(legend.position = "bottom", text = element_text(size = 10))
plot(histograms_cleaning)
if(save){
  ggsave(paste0(RESULTS_PATH, "histograms_cleaning.pdf"),
         plot = histograms_cleaning, width = 7.5, height = 3.5, units = "in")
}
  
histograms_cleaning_appendix <- ggplot(df_fullplot_appendix, aes(x = value, fill = sens_comb)) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.5, position = "identity", bins = 32) +
  labs(x=NULL, y = "Density", fill="Sensitive attribue:") +
  theme_minimal() +
  facet_wrap(~name_type, scales = "free", ncol = 5) +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  scale_fill_discrete(
    labels = c("0_0" = "Foreign men", "1_0" = "Foreign women", "0_1" = "Swiss men", "1_1" = "Swiss women")) +
  theme(legend.position = "bottom", text = element_text(size = 10))
plot(histograms_cleaning_appendix)
if(save){
  ggsave(paste0(RESULTS_PATH, "histograms_cleaning_appendix.pdf"),
         plot = histograms_cleaning_appendix, width = 9, height = 3.5, units = "in")
}


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ Benchmark policies ----------------------------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# Create functions for evaluation of assignments
# Welfare
compute_welfare <- function(Dstar, scores){
  outcomes <- mapply(function(idx, row) scores[
    row, idx], Dstar, seq_along(Dstar))
  return(mean(outcomes))
}
# Cramer's V
compute_cramerv <- function(Dstar, sens_comb){
  cramerV(table(Dstar, sens_comb))
}
# Chi2 test
compute_chi2 <- function(Dstar, sens_comb){
  round(chisq.test(table(Dstar, sens_comb))$p.value, 8)
}
# Bayes Factor
compute_BF <- function(Dstar, sens_comb){
  round(extractBF(contingencyTableBF(table(Dstar, sens_comb), sampleType = "indepMulti", fixedMargin = "cols"), onlybf=TRUE, logbf=TRUE), 8)
}
# Program frequencies
compute_prog_freq <- function(Dstar, levels=unique_D){
  prop.table(table(factor(Dstar, levels = unique_D)))
}


# Observed, blackbox and all-in-one assignments for training and evaluation sample
results_list <- list()
datasets <- list(training = training_fair_df, evaluation = evaluation_fair_df)
rows <- c("observed", "blackbox", "blackboxfair", "allin1")
cols <- c("Interpr.", "Welfare", "CramerV", "chi2 pval", "log(BF)", unique_D_ch)
for (set_name in names(datasets)) {
  df <- datasets[[set_name]]
  res <- matrix(NA, nrow = length(rows), ncol = length(cols),
                dimnames = list(rows, cols))
  # Observed assignment
  D_obs <- df[[VAR_D_NAME]]
  res["observed", ] <- c(
    FALSE,
    mean(df$outcome),
    compute_cramerv(D_obs, df$sens_comb),
    compute_chi2(D_obs, df$sens_comb),
    compute_BF(D_obs, df$sens_comb),
    compute_prog_freq(D_obs)
  )
  # Blackbox assignment
  D_blackbox <- apply(df[VAR_POLSCORE_NAME], 1, which.max)
  res["blackbox", ] <- c(
    FALSE,
    compute_welfare(D_blackbox, df[VAR_POLSCORE_NAME]),
    compute_cramerv(D_blackbox, df$sens_comb),
    compute_chi2(D_blackbox, df$sens_comb),
    compute_BF(D_blackbox, df$sens_comb),
    compute_prog_freq(D_blackbox - 1)
  )
  # Fair blackbox assignment
  D_blackboxfair <- apply(df[paste0(VAR_POLSCORE_NAME,"_mq")], 1, which.max)
  res["blackboxfair", ] <- c(
    FALSE,
    compute_welfare(D_blackboxfair, df[VAR_POLSCORE_NAME]),
    compute_cramerv(D_blackboxfair, df$sens_comb),
    compute_chi2(D_blackboxfair, df$sens_comb),
    compute_BF(D_blackboxfair, df$sens_comb),
    compute_prog_freq(D_blackboxfair - 1)
  )
  # All-in-one benchmark
  best_program <- which.max(colMeans(df[VAR_POLSCORE_NAME]))
  D_allin1 <- rep(best_program, nrow(df))
  res["allin1", ] <- c(
    TRUE,
    compute_welfare(D_allin1, df[VAR_POLSCORE_NAME]),
    0,
    1,
    log(0),
    compute_prog_freq(D_allin1 - 1)
  )
  
  results_list[[set_name]] <- as.data.frame(res)
}

# Print results
for (name in names(results_list)) {
  cat(paste0(tools::toTitleCase(name), " sample:\n"))
  print(round(results_list[[name]], 3))
  cat("\n")
}

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ Run & evaluate policy trees -------------------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# Determine tree parameters besides depth:
# split.step:  To avoid lengthy computations, we determine a maximum of 100
# evaluation points for continuous variables. Hence, we set split.step = n/100
splitstep <- round(nrow(training_df)/100, 0)
# We set the minimum node size to 0.1*(n/depth):
minnodesize <- round(0.1*(nrow(training_df)/PT_DEPTH), 0)
# Determine tree type
if(PT_DEPTH == PT_SEARCH_DEPTH){
  tree_FUN <- function(
    X, Gamma, depth = PT_DEPTH, split.step = splitstep, 
    min.node.size = minnodesize, verbose = TRUE){
    policy_tree(
      X, Gamma, depth = depth, split.step = split.step,
      min.node.size = min.node.size, verbose=verbose)
  }
}else{
  tree_FUN <- function(
    X, Gamma, depth = PT_DEPTH, search.depth = PT_SEARCH_DEPTH,
    split.step = splitstep, min.node.size = minnodesize, verbose = TRUE){
    hybrid_policy_tree(
      X, Gamma, depth = depth, search.depth = search.depth,
      split.step = split.step, min.node.size = min.node.size, verbose=verbose)
  }
}
# Run and evaluate policy trees for different adjustment scenarios
opt_tree_list <- list()
data_list <- list(
  pt_no_adjustments = list(
    exp = TRUE, As = colnames(As_org), scores = colnames(scores_org)),
  pt_adjust_A = list(
    exp = FALSE, As = colnames(As_mq), scores = colnames(scores_org)),
  pt_adjust_score = list(
    exp = TRUE, As = colnames(As_org), scores = colnames(scores_mq)),
  pt_adjust_A_score = list(
    exp = FALSE, As = colnames(As_mq), scores = colnames(scores_mq)),
  pt_adjust_Acdf = list(
    exp = FALSE, As = colnames(As_cdf), scores = colnames(scores_org)),
  pt_adjust_Acdf_score = list(
    exp = FALSE, As = colnames(As_cdf), scores = colnames(scores_mq)))
for(set_name in names(data_list)) {
  opt_tree <- tree_FUN(
    training_fair_df[data_list[[set_name]]$As],
    training_fair_df[data_list[[set_name]]$scores])
  opt_tree_list[[set_name]] <- opt_tree
  Dstar_opt <- predict(opt_tree, training_fair_df[data_list[[set_name]]$As])
  Dstar_opt_out <- predict(opt_tree, evaluation_fair_df[data_list[[set_name]]$As])
  results_list$training[set_name, ] <- c(
    data_list[[set_name]]$exp,
    compute_welfare(Dstar_opt, scores_org),
    compute_cramerv(Dstar_opt, sens_comb_org$sens_comb),
    compute_chi2(Dstar_opt, sens_comb_org$sens_comb),
    compute_BF(Dstar_opt, sens_comb_org$sens_comb),
    compute_prog_freq(Dstar_opt - 1))
  results_list$evaluation[set_name, ] <- c(
    data_list[[set_name]]$exp,
    compute_welfare(Dstar_opt_out, scores_org_out),
    compute_cramerv(Dstar_opt_out, sens_comb_org_out$sens_comb),
    compute_chi2(Dstar_opt_out, sens_comb_org_out$sens_comb),
    compute_BF(Dstar_opt_out, sens_comb_org_out$sens_comb),
    compute_prog_freq(Dstar_opt_out - 1))
}
# Print results
for (name in names(results_list)) {
  cat(paste0(tools::toTitleCase(name), " sample:\n"))
  print(round(results_list[[name]], 3))
  cat("\n")
}
# Plot the trees
lab <- mappings[!is.na(mappings[VAR_D_NAME]), 'X']
plot(opt_tree_list[['pt_no_adjustments']], leaf.labels=lab)
plot(opt_tree_list[['pt_adjust_A']], leaf.labels=lab)
plot(opt_tree_list[['pt_adjust_score']], leaf.labels=lab)
plot(opt_tree_list[['pt_adjust_A_score']], leaf.labels=lab)
plot(opt_tree_list[['pt_adjust_Acdf']], leaf.labels=lab)
plot(opt_tree_list[['pt_adjust_Acdf_score']], leaf.labels=lab)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ Run & evaluate probabilistic split trees ------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# Fit probabilistic split tree
data_list2 <- list(
  pst_adjust_A = list(exp = TRUE, adjust_scores = FALSE),
  pst_adjust_A_score = list(exp = TRUE, adjust_scores = TRUE))
for(set_name in names(data_list2)) {
  opt_tree <- prob_split_tree(
    As_org,
    scores_org,
    sens_org,
    adjust_scores=data_list2[[set_name]]$adjust_scores,
    seed=SEED,
    ties.method=TIES_METHOD,
    depth=PT_DEPTH,
    search.depth=PT_SEARCH_DEPTH,
    split.step=splitstep,
    min.node.size=minnodesize)
  opt_tree_list[[set_name]] <- opt_tree
  Dstar_opt <- predict(opt_tree, As_org, sens_org, seed=SEED)
  Dstar_opt_out <- predict(opt_tree, As_org_out, sens_org_out, seed=SEED)
  results_list$training[set_name, ] <- c(
    data_list2[[set_name]]$exp,
    compute_welfare(Dstar_opt, scores_org),
    compute_cramerv(Dstar_opt, sens_comb_org$sens_comb),
    compute_chi2(Dstar_opt, sens_comb_org$sens_comb),
    compute_BF(Dstar_opt, sens_comb_org$sens_comb),
    compute_prog_freq(Dstar_opt - 1))
  results_list$evaluation[set_name, ] <- c(
    data_list2[[set_name]]$exp,
    compute_welfare(Dstar_opt_out, scores_org_out),
    compute_cramerv(Dstar_opt_out, sens_comb_org_out$sens_comb),
    compute_chi2(Dstar_opt_out, sens_comb_org_out$sens_comb),
    compute_BF(Dstar_opt_out, sens_comb_org_out$sens_comb),
    compute_prog_freq(Dstar_opt_out - 1))
}
# Print results
for (name in names(results_list)) {
  cat(paste0(tools::toTitleCase(name), " sample:\n"))
  print(round(results_list[[name]], 3))
  cat("\n")
}
# Plot the trees
plot(opt_tree_list[['pst_adjust_A']], sens_names=VAR_S_NAME_ORD, leaf.labels=lab)
plot(opt_tree_list[['pst_adjust_A_score']], sens_names=VAR_S_NAME_ORD, leaf.labels=lab)
# Combine to one tree by manually adjusting dot string
for(set_name in names(data_list2)) {
  dot_string_total <- ""
  j <- 0
  for(i in names(opt_tree_list[[set_name]])){
    j <- j + 1
    dot_string2 <- plot(opt_tree_list[[set_name]][[i]], leaf.labels=lab)[["x"]][["diagram"]]
    dot_string2 <- gsub("\n(\\d)", paste0("\n",j,"\\1"), dot_string2)
    dot_string2 <- gsub("-> (\\d)", paste0("-> ",j,"\\1"), dot_string2)
    dot_string2 <- sub("digraph nodes { \n node [shape=box] ;\n", "", dot_string2, fixed = TRUE)
    dot_string2 <- sub("\n}", "", dot_string2, fixed = TRUE)
    dot_string2 <- gsub("[labeldistance=2.5, labelangle=45, headlabel=\"True\"]", "", dot_string2, fixed = TRUE)
    dot_string2 <- gsub("[labeldistance=2.5, labelangle=-45, headlabel=\"False\"]", "", dot_string2, fixed = TRUE)
    dot_string_total <- paste0(dot_string_total, dot_string2)
  }
  dot_string <- paste0("
      digraph combined {
        rankdir=LR;
        node [shape=box];
        top [label=\"female = 1\", shape=ellipse, style=filled, fillcolor=lightblue];
        top2 [label=\"swiss = 1\", shape=ellipse, style=filled, fillcolor=lightblue];
        top3 [label=\"swiss = 1\", shape=ellipse, style=filled, fillcolor=lightblue];
        ", dot_string_total, "
        top -> top2 [labeldistance=3, labelangle=-30, headlabel='True'];
        top -> top3 [labeldistance=3, labelangle=30, headlabel='False'];
        top2 -> ", 10, ";
        top2 -> ", 20, ";
        top3 -> ", 30, ";
        top3 -> ", 40, ";
      }
    ")
  dot_string <- gsub("qual_degree", "degree", dot_string, fixed = TRUE)
  dot_string <- gsub("past_income", "past earnings", dot_string, fixed = TRUE)
  print(DiagrammeR::grViz(dot_string))
  if(save){
    DiagrammeR::grViz(dot_string) %>% 
      DiagrammeRsvg::export_svg() %>% 
      charToRaw() %>% 
      rsvg::rsvg_pdf(paste0(RESULTS_PATH, set_name, "_extleft.pdf"))
  }
}

# Additional checks:
# Run exact version to ensure that is really exactly the same
plot(opt_tree_list[['pt_adjust_Acdf']], leaf.labels = lab)
# Transform the tree thresholds
opt_tree_org_Acdf_trans <- fairpolicytree:::transform_tree(
  tree=opt_tree_list[['pt_adjust_Acdf']],
  sens=sens_org,
  vars=As_org,
  vars_cdf=As_cdf
)
# Return predicted nodes to compare to org res
nodes_org_Acdf_trans_check <- fairpolicytree:::transform_tree(
  tree=opt_tree_list[['pt_adjust_Acdf']],
  sens=sens_org,
  vars=As_org,
  vars_cdf=As_cdf,
  return_nodes=TRUE
)
print("Distribution of predicted final nodes in org_res_trans")
print(table(nodes_org_Acdf_trans_check))
# Compare to nodes from residualized tree
nodes <- predict(opt_tree_list[['pt_adjust_Acdf']], As_cdf, type="node.id")
print("Distribution of predicted final nodes in org_res")
print(table(nodes))
# Same result!
# Check that equal for all observations
all.equal(nodes, nodes_org_Acdf_trans_check)

# Evaluate at new thresholds using exact version
idx_sens <- names(opt_tree_org_Acdf_trans)
Dstar_org_Acdf_transform_exact <- nodes_org_Acdf_trans_check*NA
for(sens in idx_sens){
  A_values <- as.numeric(unlist(strsplit(sens, "_")))
  if(length(A_values) == 1) {
    bool <- training_fair_df[VAR_S_NAME_ORD][, 1] == A_values
  } else {
    bool <- apply(training_fair_df[VAR_S_NAME_ORD][, seq_along(
      A_values)], 1, function(row) all(row == A_values))
  }
  filtered_df <- subset(training_fair_df[VAR_A_NAME_ORD], bool)
  filtered_df_Acdf <- subset(training_fair_df[paste0(VAR_A_NAME_ORD, "_cdf")], bool)
  Dstar_org_Acdf_transform_exact[bool] <- predict(
    opt_tree_org_Acdf_trans[[sens]], filtered_df, opt_tree_list[['pt_adjust_Acdf']], filtered_df_Acdf)
}
# Exact method gives exactly the same result as pt_adjust_Acdf
print(sum(predict(opt_tree_list[['pt_adjust_Acdf']], As_cdf) != Dstar_org_Acdf_transform_exact))

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ Export results table to latex -----------------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
for (name in names(results_list)) {
  sample_txt <- if(name=="evaluation") "out-of-sample" else "in-sample"
  results_export_out <- results_list[[name]]
  # Select columns to export
  cols_to_export <- c(
    "observed",
    "blackbox",
    "blackboxfair",
    "allin1",
    "pt_no_adjustments",
    "pt_adjust_A",
    "pt_adjust_score",
    "pt_adjust_A_score",
    "pst_adjust_A",
    "pst_adjust_A_score" )
  results_export_out <- results_export_out[cols_to_export,]
  policies <- data.frame(Policy = c(
    "Observed",
    "Blackbox",
    "Blackbox fair",
    "All in one",
    "No adjustments",
    "Adjust $A_i$",
    "Adjust $\\Gamma_{d,i}$",
    "Adjust $A_i$ and $\\Gamma_{d,i}$",
    "Adjust $A_i$",
    "Adjust $A_i$ and $\\Gamma_{d,i}$"
  ))
  results_export_out <- cbind(policies, data.frame(results_export_out, row.names=NULL))
  colnames(results_export_out) <- c(
    "\\textbf{Policy}", "\\textbf{Interp.}", "\\textbf{Welfare}", "CramerV",
    "p-value", "log(BF)", "NP", "JS", "VC", "CC", "LC", "EP")
  
  # Manually adjust columns
  results_export_out <- results_export_out %>%
    mutate(`\\textbf{Welfare}` = round(as.numeric(`\\textbf{Welfare}`), 3),
           `CramerV` = round(as.numeric(`CramerV`), 3),
           `p-value` = round(as.numeric(`p-value`), 3),
           `log(BF)` = round(as.numeric(`log(BF)`), 3))
  #Transform treatment shares to percentages
  results_export_out <- results_export_out %>%
    mutate(across(
      .cols = (ncol(results_export_out)-nunique_D+1):ncol(results_export_out),
      .fns = ~ ifelse(is.na(as.numeric(.)), NA, sprintf("%.1f\\%%", round(as.numeric(.) * 100, 2)))
    ))
  results_export_out[["log(BF)"]] <- format(round(results_export_out[["log(BF)"]], 0), nsmall = 0)
  note = paste(
    "\\\\setstretch{1}\\\\scriptsize \\\\textit{Notes:} This table presents measures of interpretability,",
    "welfare, and fairness for various policy types.",
    "The column labeled \\\\textit{Interp.} indicates whether the policy is interpretable. The \\\\textit{Welfare} column reports",
    "the mean potential outcome under the assignments of the respective policy. The next three columns",
    "display fairness metrics: Cramer's V, its associated p-value, and the logarithm of the Bayes Factor.",
    "The remaining columns report the resulting program shares under each policy, with \\\\textit{NP} = No Program,",
    "\\\\textit{JS} = Job Search, \\\\textit{VC} = Vocational Course, \\\\textit{CC} = Computer Course,",
    "\\\\textit{LC} = Language Course, \\\\textit{EP} = Employment Program."
  )
  if(name=="evaluation"){
    note <- paste(note, "Statistics are computed out-of-sample using data not used for estimating policy scores or training the policy tree.")
  }
  results_export_out_latex <- kable(results_export_out, format = "latex", escape = FALSE, booktabs=T,
                                    caption = paste0("Welfare-Fairness-Interpretability trade-off for different policies (", sample_txt,")"), 
                                    label = paste0("tab:main_results_", substr(sample_txt,1,2)), align='llrrrrrrrrrr', full_width = T) %>%
    kable_styling(latex_options = c("HOLD_position"), font_size = 8) %>%
    add_header_above(c(" " = 3, "Fairness" = 3, "Program shares" = 6), bold = T, line=T) %>%
    pack_rows("Benchmark policies", 1, 4) %>%
    pack_rows(paste0("Optimal policy tree (depth ", PT_DEPTH, ")"), 5, 8) %>%
    pack_rows(paste0("Probabilistic split tree (depth ", PT_DEPTH, ")"), 9, 10) %>%
    # column_spec(6, width = "2cm") %>%
    footnote(general = note, footnote_as_chunk=T, general_title="", threeparttable = T, escape = FALSE)
  writeLines(results_export_out_latex, paste0(RESULTS_PATH, "main_results_", substr(sample_txt,1,2), ".tex"))
}

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ Partial adjustments ---------------------------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# Get grid from 0 to 1 by 0.1 steps
# get weighted combinations of score and As
# Run policy trees
# Compute stats and plot
weights <- seq(0,1,0.1)
inv_weights <- 1 - weights
As_weight <- mapply(function(w, iw) round(As_mq * w + As_org * iw, 0), weights, inv_weights, SIMPLIFY = FALSE)
As_weight_out <- mapply(function(w, iw) round(As_mq_out * w + As_org_out * iw, 0), weights, inv_weights, SIMPLIFY = FALSE)
scores_weight <- mapply(function(w, iw) scores_mq * w + scores_org * iw, weights, inv_weights, SIMPLIFY = FALSE)
As_org_list <- replicate(length(weights), As_org, simplify = FALSE)
As_org_out_list <- replicate(length(weights), As_org_out, simplify = FALSE)
scores_org_list <- replicate(length(weights), scores_org, simplify = FALSE)

scenarios <- list(
  clean_As = list(As = As_weight, scores = scores_org_list, As_out = As_weight_out),
  clean_Scores = list(As = As_org_list, scores = scores_weight, As_out = As_org_out_list),
  clean_Scores_As = list(As = As_weight, scores = scores_weight, As_out = As_weight_out)
)

rows <- list()
opt_tree_weight <- list()
for(scen in names(scenarios)){
  As_list <- scenarios[[scen]]$As
  Scores_list <- scenarios[[scen]]$scores
  As_out_list <- scenarios[[scen]]$As_out
  opt_tree_weight[[scen]] <- list()
  for(i in seq_along(weights)){
    print(paste0(scen, ": ", i))
    opt_tree_weight[[scen]][[i]] <- tree_FUN(As_list[[i]], Scores_list[[i]])
    Dstar <- predict(opt_tree_weight[[scen]][[i]], As_out_list[[i]])
    rows[[length(rows) + 1]] <- data.frame(
      Weight = weights[i], 
      Metric='Welfare', 
      Scenario=scen, 
      Value=compute_welfare(Dstar, scores_org_out))
    rows[[(length(rows) + 1)]] <- data.frame(
      Weight = weights[i], 
      Metric='CramerV', 
      Scenario=scen, 
      Value=compute_cramerv(Dstar, sens_comb_org_out$sens_comb)[[1]])
  }
}
results_partial <- do.call(rbind, rows)

# Plot
p1 <- ggplot(results_partial[results_partial$Metric == "Welfare", ], aes(x = Weight, y = Value, color = Scenario, linetype= Scenario)) +
  geom_line(linewidth=0.7) +
  labs(title = "Welfare", x = "Weight of adjusted variable", color = "Scenario:", linetype= "Scenario:") +
  ylim(17.8, 18) + 
  theme_minimal() +
  scale_color_discrete(labels = c(expression(paste("Adjust ", A[i])), expression(paste("Adjust ", Gamma["d,i"])), expression(paste("Adjust ", A[i], " and ", Gamma["d,i"])))) +
  scale_linetype_discrete(labels = c(expression(paste("Adjust ", A[i])), expression(paste("Adjust ", Gamma["d,i"])), expression(paste("Adjust ", A[i], " and ", Gamma["d,i"]))))
p2 <- ggplot(results_partial[results_partial$Metric == "CramerV", ], aes(x = Weight, y = Value, color = Scenario, linetype= Scenario)) +
  geom_line(linewidth=0.7) +
  labs(title = "Fairness (Cramer's V)", x = "Weight of adjusted variable", color = "Scenario:", linetype= "Scenario:") +
  ylim(0, 0.3) + 
  theme_minimal() +
  scale_color_discrete(labels = c(expression(paste("Adjust ", A[i])), expression(paste("Adjust ", Gamma["d,i"])), expression(paste("Adjust ", A[i], " and ", Gamma["d,i"])))) +
  scale_linetype_discrete(labels = c(expression(paste("Adjust ", A[i])), expression(paste("Adjust ", Gamma["d,i"])), expression(paste("Adjust ", A[i], " and ", Gamma["d,i"]))))
plot_combined <- (p1 + p2) + plot_layout(guides = "collect") 
if(save){
  ggsave(paste0(RESULTS_PATH, "partial_adjustment.pdf"),
         plot = plot_combined, width = 7.5, height = 2.5, units = "in")
}



# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# ------------ Check winners and loosers ---------------------------------------
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# Prepare data
individual_welfare <- function(Dstar, scores) {
  scores[cbind(seq_along(Dstar), Dstar)]
}
data_df <- cbind(
  evaluation_df[, columns$X_ord[columns$X_ord > 0]],
  fastDummies::dummy_cols(
    data.frame(lapply(evaluation_df[, columns$X_unord[columns$X_unord > 0]], as.character)),
    remove_selected_columns = TRUE
  ),
  welfare_adjusted = individual_welfare(
    predict(tree = opt_tree_list[["pst_adjust_A"]], A = As_org_out, sens = sens_org_out),
    scores_org_out
  ),
  welfare_unadjusted = individual_welfare(
    predict(opt_tree_list[["pt_no_adjustments"]], As_org_out),
    scores_org_out
  )
)
# Compute the difference between actual and reference welfare.
data_df$welfare_diff <- data_df$welfare_adjusted - data_df$welfare_unadjusted
welfare_diff_matrix <- matrix(data_df$welfare_diff, ncol = 1)

# Determine optimal number of clusters using Silhouette score
dist_matrix <- stats::dist(welfare_diff_matrix)
min_cluster_size <- ceiling(nrow(data_df) / 100)
sil_width <- sapply(2:8, function(k){
  km <- kmeans(welfare_diff_matrix, centers = k, nstart = 10)
  ss <- silhouette(km$cluster, dist_matrix)
  if(min(km$size) > min_cluster_size){
    mean(ss[, 3])
  }else{
    0
  }
})
n_clusters <- which.max(sil_width) + 1
# Get clusters ordered by cluster mean
km <- kmeans(welfare_diff_matrix, centers = n_clusters, nstart = 10)
ordered_clusters <- order(km$centers)
cluster_map <- setNames(seq_along(ordered_clusters), ordered_clusters)
data_df$cluster_ids <- cluster_map[as.character(km$cluster)]
# Get variables means
df_grouped_means <- data_df %>%
  group_by(cluster_ids) %>%
  summarise(n = n(), across(where(is.numeric),\(x) mean(x, na.rm = TRUE)))  %>%
  pivot_longer(cols = -cluster_ids, names_to = "variable", values_to = "mean") %>%
  pivot_wider(names_from = cluster_ids, values_from = mean) %>%
  arrange(factor(variable, levels = c("welfare_diff", setdiff(variable, "welfare_diff")))) %>% 
  mutate(across(where(is.numeric), round, 2))

# Function to generale cluster labels
generate_cluster_labels <- function(means) {
  labels <- character(length(means))
  gains <- means > 0
  losses <- means < 0
  zeros <- means == 0
  intensity_labels <- c("Strong", "Moderate", "Mild", "Slight")
  # Gains
  if(sum(gains) > 0){
    ranks <- rank(-means[gains], ties.method = "first")
    intensity <- if(sum(gains) > 1) paste(intensity_labels[seq_along(ranks)], "Gain") else "Gain"
    labels[gains] <- intensity[order(ranks)]
  }
  # Losses
  if(sum(losses) > 0){
    ranks <- rank(means[losses], ties.method = "first")
    intensity <- if(sum(losses) > 1) paste(intensity_labels[seq_along(ranks)], "Loss") else "Loss"
    labels[losses] <- intensity[order(ranks)]
  }
  # No change
  labels[zeros] <- "No Change"
  return(labels)
}

colnames(df_grouped_means) <- c("Variable", generate_cluster_labels(
  round(as.numeric(subset(df_grouped_means, variable == 'welfare_diff')[, -1]),2)))

note <- paste(
  "\\\\setstretch{1}\\\\scriptsize \\\\textit{Notes:}",
  "The table shows mean values of variables within clusters obtained via k-means clustering.",
  "Clustering is based on the difference in welfare between optimal policy trees with and without",
  "adjustments to the decision variables. The number of clusters is determined in a data-driven",
  "way by the Silhouette score and the minimum required cluster size is set to 1\\\\% of the observations.")


#Selection short table
vars_short <- c(
  "welfare_diff",
  "n",
  "female",
  "swiss",
  "age",
  "past_income",
  "qual_degree",
  "conton_moth_tongue",
  "emp_share_last_2yrs",
  "emp_spells_5yrs",
  "married",
  "other_mother_tongue",
  "ue_spells_last_2yrs",
  "city_1",
  "city_2",
  "city_3"
)

# Adjust variable names
map_varnames <- c(
  "welfare_diff" = "Welfare difference",
  "n" = "Number of observations in cluster",
  "female" = "Job seeker is female",
  "swiss" = "Swiss citizen",
  "age" = "Age of job seeker",
  "canton_moth_tongue" = "Mother tongue in canton's language",
  "cw_age" = "Age of caseworker",
  "cw_cooperative" = "Caseworker cooperative",
  "cw_educ_above_voc" = "Caseworker education: above vocational training",
  "cw_educ_tertiary" = "Caseworker education: tertiary track" ,
  "cw_female" = "Caseworker female",
  "cw_missing" = "Indicator for missing caseworker characteristics",
  "cw_own_ue" = "Caseworker has own unemployemnt experience",
  "cw_tenure" = "Caseworker job tenure in years",
  "cw_voc_degree" = "Caseworker education: vocational training degree",
  "emp_share_last_2yrs" = "Fraction months employed in last 2 years",
  "emp_spells_5yrs" = "Employment spells in last 5 years",
  "employability" = "Employability as assessed by the caseworker",
  "foreigner_b" = "Foreigner with temporary permit (B permit)",
  "foreigner_c" = "Foreigner with permanent permit (C permit)",
  "gdp_pc" = "Cantonal GDP per capita (in CHF 10,000)",
  "married" = "Job seeker is married",
  "other_mother_tongue" = "Mother tongue other than German, French, Italian",
  "past_income" = "Earnings in CHF before unemployment",
  "ue_cw_allocation1" = "Allocation of unemployed to caseworkers: by industry",
  "ue_cw_allocation2" = "Allocation of unemployed to caseworkers: by occupation",
  "ue_cw_allocation3" = "Allocation of unemployed to caseworkers: by age",
  "ue_cw_allocation4" = "Allocation of unemployed to caseworkers: by employability",
  "ue_cw_allocation5" = "Allocation of unemployed to caseworkers: by region",
  "ue_cw_allocation6" = "Allocation of unemployed to caseworkers: other",
  "ue_spells_last_2yrs" = "Unemployment spells in last 2 years",
  "unemp_rate" = "Cantonal unemployment rate (in %)",
  "city_3" = "Lives in big city",
  "city_2" = "Lives in medium city",
  "city_1" = "Lives in no city",
  "prev_job_sec_cat_0" = "Sector of last job: tertiary sector",
  "prev_job_sec_cat_1" = "Sector of last job: secondary sector",
  "prev_job_sec_cat_2" = "Sector of last job: missing sector",
  "prev_job_sec_cat_3" = "Sector of last job: primary sector",
  "prev_job_0" = "Previous job: skilled worker",
  "prev_job_1" = "Previous job: manager",
  "prev_job_2" = "Previous job: unskilled worker",
  "prev_job_3" = "Previous job: self-employed",
  "qual_0" = "Qualification: with degree",
  "qual_1" = "Qualification: semiskilled",
  "qual_2" = "Qualification: unskilled",
  "qual_3" = "Qualification: no degree",
  "qual_degree" = "Qualification: with degree"
)

# Loop through numeric columns
df_grouped_means <- as.data.frame(df_grouped_means)
for (col in names(df_grouped_means)[-1]) {
  df_grouped_means[[col]] <- ifelse(
    df_grouped_means$Variable %in% c("n", "past_income"),
    formatC(df_grouped_means[[col]], format = "f", digits = 0),
    formatC(df_grouped_means[[col]], format = "f", digits = 2)
  )
}


# Create short table
df_grouped_means_short <- df_grouped_means[df_grouped_means$Variable %in% vars_short,]
df_grouped_means_short <- df_grouped_means_short %>%
  mutate(Variable = factor(Variable, levels = vars_short)) %>%
  arrange(Variable) %>%
  mutate(Variable = as.character(Variable))
df_grouped_means_short$Variable <- map_varnames[df_grouped_means_short$Variable]

# Create long table
df_grouped_means_long <- df_grouped_means[!df_grouped_means$Variable %in% c(
  "welfare_adjusted", "welfare_unadjusted"), ]
match_idx <- match(df_grouped_means_long$Variable, vars_short)
original_order <- seq_len(nrow(df_grouped_means_long))
df_grouped_means_long <- df_grouped_means_long[order(is.na(match_idx), match_idx, original_order), ]
df_grouped_means_long$Variable <- map_varnames[df_grouped_means_long$Variable]
rownames(df_grouped_means_long) <- NULL

if(save){
  kmeans_export_latex_short <- kable(
    df_grouped_means_short, format = "latex", escape = T, booktabs=T, linesep = "",
    caption = "Covariate means of winners and losers from fairness-based reassignment", 
    label = "tab:kmeans_short", align=paste0("l", strrep("r", n_clusters)), full_width = T,
    latex_header_includes = c("\\renewcommand{\\arraystretch}{2}"))%>%
    kable_styling(latex_options = c("HOLD_position"), font_size = 8) %>%
    add_header_above(c(" " = 1, "Cluster (sorted by welfare change)" = n_clusters), bold = T, line=T) %>%
    footnote(general = note, footnote_as_chunk=T, general_title="", threeparttable = T, escape = FALSE) %>%
    row_spec(c(2,4,7), hline_after = TRUE) %>%
    column_spec(2:(n_clusters+1), width = "2cm")
  kmeans_export_latex_short <- gsub("midrule\\\\", "midrule", kmeans_export_latex_short, fixed = TRUE)
  writeLines(kmeans_export_latex_short, paste0(RESULTS_PATH, "kmeans_short.tex"))
  
  kmeans_export_latex_long <- kable(
    df_grouped_means_long, format = "latex", escape = T, booktabs=T, linesep = "",
    caption = "Covariate means of winners and losers from fairness-based reassignment (all covariates)", 
    label = "tab:kmeans_long", align=paste0("l", strrep("r", n_clusters)), full_width = T,
    latex_header_includes = c("\\renewcommand{\\arraystretch}{2}"))%>%
    kable_styling(latex_options = c("HOLD_position"), font_size = 8) %>%
    add_header_above(c(" " = 1, "Cluster (sorted by welfare change)" = n_clusters), bold = T, line=T) %>%
    footnote(general = note, footnote_as_chunk=T, general_title="", threeparttable = T, escape = FALSE) %>%
    row_spec(c(2,4,7), hline_after = TRUE) %>%
    column_spec(2:(n_clusters+1), width = "2cm")
  kmeans_export_latex_long <- gsub("midrule\\\\", "midrule", kmeans_export_latex_long, fixed = TRUE)
  writeLines(kmeans_export_latex_long, paste0(RESULTS_PATH, "kmeans_long.tex"))
}
