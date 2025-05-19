#' Marginal Quantile-Adjustment for Fair and Interpretable Policy Learning
#'
#' Computes fairness-adjusted variables by MQ-adjustment
#'
#' @param vars Numeric matrix or data.frame of variables to be adjusted (observations in rows, variables in columns).
#' @param sens Matrix or data.frame of sensitive attributes. Must have the same number of rows as `vars`.
#' @param seed Integer scalar for reproducible random tie-breaking.
#' @param ties.method Character string for ranking ties. One of "random", "average", "first", "last", "max", "min".
#'
#' @return A list with two data.frames:
#'   \describe{
#'     \item{vars_cdf}{CDF-adjusted variables denoted with suffix `_cdf`.}
#'     \item{vars_mq}{MQ-adjusted variables denoted with suffix `_mq`.}
#'   }
#'
#' @export
mq_adjustment <- function(
    vars,
    sens,
    seed = 123456,
    ties.method = "random"
) {
  #-- Input checks --#
  # Seed
  if (!is.numeric(seed) || length(seed) != 1) {
    stop("`seed` must be a single numeric value.")
  }
  # Ties method
  valid_methods <- c("random", "average", "first", "last", "max", "min")
  if (!is.character(ties.method) || length(ties.method) != 1 ||
      !(ties.method %in% valid_methods)) {
    stop(sprintf("`ties.method` must be one of: %s.", paste(valid_methods, collapse = ", ")))
  }
  # Variables to be adjusted
  if (!(is.matrix(vars) || is.data.frame(vars))) {
    stop("`vars` must be a matrix or data.frame.")
  }
  vars <- as.data.frame(vars)
  n <- nrow(vars)
  # Sensitive attributes
  if (!(is.matrix(sens) || is.data.frame(sens))) {
    stop("`sens` must be a matrix or data.frame.")
  }
  sens <- as.data.frame(sens, stringsAsFactors = FALSE)
  if (nrow(sens) != n) {
    stop("`sens` must have the same number of rows as `vars`.")
  }
  #-- Main computation --#
  set.seed(seed)
  no_of_vars <- ncol(vars)
  no_of_obs <- nrow(vars)
  vars_mq <- vars
  vars_cdf  <- vars
  group_keys <- do.call(paste, c(sens, sep = "\r"))
  groups <- split(seq_len(n), group_keys)
  # Compute empirical CDF (relative rank) within each sensitive group
  for (col in seq_len(no_of_vars)) {
    vars_col <- vars[[col]]
    quantiles <- numeric(no_of_obs)
    for (key in names(groups)) {
      idx <- groups[[key]]
      quantiles[idx] <- rank(vars_col[idx], ties.method=ties.method)/length(idx)
    }
    vars_cdf[[col]] <- quantiles
    vars_col_sorted <- sort(vars_col)
    indices <- rank(quantiles, ties.method = ties.method)
    vars_mq[[col]] <- vars_col_sorted[indices]
  }
  colnames(vars_mq) <- paste0(colnames(vars_mq), "_mq")
  colnames(vars_cdf)  <- paste0(colnames(vars_cdf),  "_cdf")
  return(list(vars_cdf = vars_cdf, vars_mq = vars_mq))
}


#' Simulate Artificial Fairness Data
#'
#' Generates artificial data with:
#' - Two binary sensitive attributes (correlated)
#' - One binary and one continuous decision variable (both correlated with sensitive attributes)
#' - Two continuous policy score variables (correlated with all other variables)
#'
#' @param n Integer. Number of observations to generate. Default is 1000.
#' @param seed Integer. Random seed for reproducibility. Default is 123456.
#'
#' @return A list with three data.frames:
#'   \describe{
#'     \item{sens}{Data frame with two binary sensitive attributes.}
#'     \item{decision}{Data frame with one binary and one continuous decision variable.}
#'     \item{scores}{Data frame with two continuous policy score variables.}
#'   }
#'
#' @export
simulate_fairness_data <- function(n = 1000, seed = 123456) {
  #-- Input checks --#
  if (!is.numeric(n) || length(n) != 1 || n <= 0 || n != as.integer(n)) {
    stop("`n` must be a single positive integer.")
  }
  if (!is.numeric(seed) || length(seed) != 1 || seed != as.integer(seed)) {
    stop("`seed` must be a single integer.")
  }
  #-- Main computation --#
  set.seed(seed)
  # Step 1: Generate two correlated binary sensitive attributes
  sens1 <- rbinom(n, 1, 0.5)
  sens2 <- rbinom(n, 1, 0.4 + 0.3 * sens1)  # sens2 correlated with sens1
  sens <- data.frame(sens1 = sens1, sens2 = sens2)
  # Step 2: Decision variables (correlated with sensitive attributes)
  logit_dec_bin <- -0.5 + 0.8 * sens1 + 0.6 * sens2
  prob_dec_bin <- 1 / (1 + exp(-logit_dec_bin))
  dec_bin <- rbinom(n, 1, prob_dec_bin)
  dec_cont <- 0.3 * sens1 + 0.5 * sens2 + rnorm(n)
  decision <- data.frame(dec_bin = dec_bin, dec_cont = dec_cont)
  # Step 3: Policy score variables (continuous, correlated with all above)
  X <- as.matrix(cbind(sens, decision))
  score0 <- X %*% c(0.4, -0.6, 0.5, 0.2) + rnorm(n, sd = 0.5)
  score1 <- X %*% c(-0.2, 0.3, 0.6, -0.4) + 1 + rnorm(n, sd = 0.5)
  scores <- data.frame(score0 = as.numeric(score0), score1 = as.numeric(score1))
  return(list(sens = sens, decision = decision, scores = scores))
}


#' Conditional Quantile Adjustment for Fair Interpetable Policy Learning
#'
#' Inverts a target CDF value (`eval`) back to the original variable scale,
#' separately within each sensitive group.
#'
#' @param eval Numeric scalar in [0,1]: the target quantile to invert.
#' @param var Numeric vector, or single-column matrix/data.frame of the original values.
#' @param var_cdf
#'   Numeric vector, or single-column matrix/data.frame of the CDF-transformed values
#'   (same length as `var`).
#' @param sens
#'   Vector, matrix, or data.frame of one or more sensitive attributes
#'   (must have same number of rows as `var`).
#'
#' @return A data.frame with one row per unique sensitive combination,
#'   containing:
#'   - the sensitive attributes,
#'   - `eval` (the input quantile),
#'   - the inverted value on the original scale at that quantile.
#'
#' @keywords internal
cq_adjustment <- function(
    eval,
    var,
    var_cdf,
    sens){
  ##-- Input checks --##
  if (!is.numeric(eval) || length(eval) != 1 || is.na(eval) || eval < 0 || eval > 1) {
    stop("`eval` must be a single number in [0,1].")
  }
  # coerce var to numeric vector
  var_vec <- if (is.data.frame(var) || is.matrix(var)) {
    if (ncol(var) != 1) stop("`var` must have exactly one column.")
    as.numeric(var[,1])
  } else if (is.numeric(var)) {
    as.numeric(var)
  } else stop("`var` must be numeric vector or 1-col data.")
  # coerce var_cdf to numeric vector
  cdf_vec <- if (is.data.frame(var_cdf) || is.matrix(var_cdf)) {
    if (ncol(var_cdf) != 1) stop("`var_cdf` must have exactly one column.")
    as.numeric(var_cdf[,1])
  } else if (is.numeric(var_cdf)) {
    as.numeric(var_cdf)
  } else stop("`var_cdf` must be numeric vector or 1-col data.")
  n <- length(var_vec)
  if (length(cdf_vec) != n) stop("`var` and `var_cdf` must have the same length.")
  # coerce sens to data.frame
  sens_df <- if (is.vector(sens) && !is.list(sens)) {
    data.frame(sens1 = as.character(sens), stringsAsFactors = FALSE)
  } else if (is.matrix(sens) || is.data.frame(sens)) {
    df <- as.data.frame(sens, stringsAsFactors = FALSE)
    # ensure all columns are character (so paste works predictably)
    as.data.frame(lapply(df, as.character), stringsAsFactors = FALSE)
  } else {
    stop("`sens` must be a vector, matrix, or data.frame.")
  }
  if (nrow(sens_df) != n) stop("`sens` must have the same number of rows as `var`.")
  # range check
  if (eval < min(cdf_vec, na.rm = TRUE) || eval > max(cdf_vec, na.rm = TRUE)) {
    stop("`eval` is outside the range of `var_cdf`.")
  }

  ##-- Grouping setup --##
  # build a unique key per row by pasting all sens columns
  keys   <- apply(sens_df, 1, paste, collapse = "\r")
  groups <- split(seq_len(n), keys)
  # recover one row per unique combination
  combos <- unique(sens_df)

  ##-- Prepare result frame --##
  result <- combos
  # use the original var_cdf column name if available, else "eval"
  cdf_name <- if (!is.null(colnames(var_cdf))) colnames(var_cdf)[1] else "eval"
  var_name <- if (!is.null(colnames(var)))   colnames(var)[1]   else "value"
  result[[cdf_name]] <- eval
  result[[var_name]] <- NA_real_

  ##-- Inversion loop --##
  for (i in seq_len(nrow(combos))) {
    # identify this group's key
    this_key <- paste(as.character(combos[i, , drop = FALSE]), collapse = "\r")
    idx      <- groups[[this_key]]
    xi       <- var_vec[idx]
    ci       <- cdf_vec[idx]

    # lower/upper bounds
    if (eval < min(ci, na.rm = TRUE)) {
      result[[var_name]][i] <- -Inf; next
    }
    if (eval > max(ci, na.rm = TRUE)) {
      result[[var_name]][i] <-  Inf; next
    }
    # find nearest below and above
    lower_i <- which(ci == max(ci[which(ci <= eval)]))
    upper_i <- which(ci == min(ci[which(ci >= eval)]))

    if (lower_i == upper_i) {
      result[[var_name]][i] <- xi[lower_i]
    } else {
      x0 <- ci[lower_i]
      y0 <- xi[lower_i]
      x1 <- ci[upper_i]
      y1 <- xi[upper_i]
      result[[var_name]][i] <- y0 + (y1 - y0)*(eval - x0)/(x1 - x0)
    }
  }
  return(result)
}



#' Transform a Policy Tree Using Fairness Adjustments
#'
#' Adjusts a fitted policytree object to align splitting thresholds with values
#' derived from conditional quantile adjustment.
#'
#' @param tree A policy tree fitted using the `policytree` package.
#' @param sens A matrix or data.frame of sensitive attributes.
#' @param vars A matrix or data.frame of decision variables (same as used to fit the original tree).
#' @param vars_cdf A matrix or data.frame of CDF-adjusted versions of `vars` (same dimensions).
#' @param return_nodes Logical. If TRUE, returns node IDs instead of the adjusted tree. Default is FALSE.
#'
#' @return A list of transformed tree objects, one per sensitive group, or a vector of node IDs if `return_nodes = TRUE`.
#'
#' @keywords internal
transform_tree <- function(
    tree,
    sens,
    vars,
    vars_cdf,
    return_nodes=FALSE){
  ##-- Input checks and coercion --##
  if (!inherits(tree, "policy_tree")) {
    stop("`tree` must be a fitted policy tree (class 'policy_tree').")
  }
  if (is.matrix(sens)) sens <- as.data.frame(sens, stringsAsFactors = FALSE)
  if (!is.data.frame(sens)) stop("`sens` must be a data.frame.")

  if (is.matrix(vars)) vars <- as.data.frame(vars)
  if (!is.data.frame(vars)) stop("`vars` must be a data.frame.")

  if (is.matrix(vars_cdf)) vars_cdf <- as.data.frame(vars_cdf)
  if (!is.data.frame(vars_cdf)) stop("`vars_cdf` must be a data.frame.")

  if (nrow(sens) != nrow(vars) || nrow(vars) != nrow(vars_cdf)) {
    stop("`sens`, `vars`, and `vars_cdf` must have the same number of rows.")
  }
  if (ncol(vars) != ncol(vars_cdf)) {
    stop("`vars`, and `vars_cdf` must have the same number of columns.")
  }
  # Get unique combinations of sensitive attributes
  sens_comb <- apply(sens, 1, paste, collapse = "_")
  idx_sens <- sort(unique(sens_comb))
  # Get tree array of original tree
  ta <- tree[["_tree_array"]]
  # Get number of splits
  n_splits <- sum(ta[,1] > 0)
  # Generate list of trees for each sensitive combination
  tree_trans <- list()
  dft <- list()
  for(s in idx_sens){
    tree_trans[[s]] <- tree
    class(tree_trans[[s]]) <- "prob_split_tree"
    # Translate threshold variables
    tree_trans[[s]][["columns"]] <- sapply(
      tree_trans[[s]][["columns"]], function(x) sub(
        "_[^_]*$", "", x))
    # Add probability column to tree array
    tree_trans[[s]][["_tree_array"]] <- cbind(tree_trans[[s]][[
      "_tree_array"]], rep(1, nrow(tree_trans[[s]][["_tree_array"]])))
    tree_trans[[s]][["probabilistic_splits"]] <- FALSE
    # Generate tables to save leaf ids
    dft[[s]] <- data.frame(matrix(FALSE, nrow=nrow(vars[sens_comb==s, , drop = FALSE]), ncol=nrow(ta)))
    colnames(dft[[s]]) <- seq(nrow(ta))
    dft[[s]][, 1] <- TRUE
  }
  # Create list to save transformations
  trans <- list()
  # Loop over splits of original tree and adjust new trees
  for (row in seq_len(nrow(ta))){
    if (ta[row, 1] > 0){
      # Define splitting variable and threshold
      splitvar <- tree[["columns"]][ta[row, 1]]
      splitvar_org <- sub("_[^_]*$", "", splitvar)
      splitvalue_fair <- ta[row, 2]
      left_child <- ta[row, 3]
      right_child <- ta[row, 4]
      # APPROXIMATION
      trans[[row]] <- cq_adjustment(
        eval=splitvalue_fair,
        var=vars[splitvar_org],
        var_cdf=vars_cdf[splitvar],
        sens=sens)
      rownames(trans[[row]]) <- apply(
        trans[[row]][colnames(sens)], 1, paste, collapse = "_")
      # Update dft
      trans[[row]]$share_left <- NA
      for (s in idx_sens){
        splitvalue_org <- trans[[row]][s, splitvar_org]
        # Get ids of current leaf and vars of splitting variable
        ids_current_leaf <- which(dft[[s]][, row])
        vars_sens <- vars[sens_comb==s,,drop=FALSE]
        row.names(vars_sens) <- NULL
        vars_current_leaf <- vars_sens[ids_current_leaf, splitvar_org]
        vars_cdf_sens <- vars_cdf[sens_comb==s,,drop=FALSE]
        row.names(vars_cdf_sens) <- NULL
        vars_cdf_current_leaf <- vars_cdf_sens[ids_current_leaf, splitvar]
        # Find ties
        dec <- ifelse(vars_current_leaf < splitvalue_org, -1, ifelse(vars_current_leaf == splitvalue_org, 0, 1))
        # Check share below fair threshold among ties
        share_left <- mean(vars_cdf_current_leaf[dec == 0] <= splitvalue_fair)
        # assign -1 and 1 accordingly to keep same result
        dec[dec == 0] <- (vars_cdf_current_leaf[dec == 0] > splitvalue_fair)*2-1
        # Determine ids of left and right child
        go_left <- ids_current_leaf[which(dec <= 0)]
        go_right <- ids_current_leaf[which(dec > 0)]
        #Alternative
        # go_left <- ids_current_leaf[which(vars_cdf_current_leaf <= splitvalue_fair)]
        go_right <- ids_current_leaf[which(vars_cdf_current_leaf > splitvalue_fair)]
        # Update dft
        dft[[s]][go_left, left_child] <- TRUE
        dft[[s]][go_right, right_child] <- TRUE
        # In the tree object we need to update 'nodes', '_tree_array 'and 'columns'
        tree_trans[[s]][['nodes']][[row]]$split_value <- splitvalue_org
        tree_trans[[s]][["_tree_array"]][row, 2] <- splitvalue_org
        tree_trans[[s]][["_tree_array"]][row, 5] <- ifelse(is.na(share_left), 1, share_left)
        # Save share_left
        trans[[row]][s, 'share_left'] <- ifelse(is.na(share_left), 1, share_left)
        # Update prob indicator
        if(!is.na(share_left)){
          tree_trans[[s]][["probabilistic_splits"]] <- TRUE
        }
      }
    }
  }
  # Check if prediction before and after transformation equal
  nodes_fair <- predict(tree, vars_cdf, type="node.id")
  final <- which(ta[, 1] < 0)
  nodes_trans <- nodes_fair*0
  for (s in idx_sens){
    nodes_trans[sens_comb==s] <- as.integer(colnames(dft[[s]][final])[max.col(dft[[s]][final])])
  }
  if(all.equal(nodes_trans, nodes_fair) != TRUE){
    warning("Warning: Transformation failed. Nodes in transformed tree don't exactly match fair tree.")
  }
  if(!return_nodes){
    class(tree_trans) <- "prob_split_tree_list"
    return(tree_trans)
  }else{
    return(nodes_trans)
  }
}


#' Predict from a List of Probabilistic Split Trees
#'
#' Makes group-specific predictions using a list of probabilistic policy trees,
#' each fitted for a unique combination of sensitive attributes.
#'
#' @param tree_list A named list of fitted probabilistic split trees (e.g., from [prob_split_tree()]),
#' where names are underscore-separated group identifiers (e.g., `"0_1"`).
#' @param A A matrix or data.frame of decision variables.
#' @param sens A data.frame or matrix of sensitive attributes used for fairness adjustment.
#' (must match the naming in `tree_list`).
#' @param type Character. One of `"action.id"` (default) or `"leaf.id"`. Determines the type of prediction returned.
#' @param seed Integer. Random seed used to resolve probabilistic splits when exact info is missing. Default is 123456.
#'
#' @return A vector of predicted actions or leaf IDs, one per row of `A`.
#'
#' @method predict prob_split_tree_list
#'
#' @export
predict.prob_split_tree_list <- function(
    tree_list, A, sens, type = "action.id", seed = 123456) {
  ##-- Input checks --##
  if (!inherits(tree_list, "prob_split_tree_list")) {
    stop("`tree_list` must be of class 'prob_split_tree_list'.")
  }

  if (!(is.matrix(A) || is.data.frame(A))) {
    stop("`A` must be a matrix or data.frame.")
  }
  if (!(is.matrix(sens) || is.data.frame(sens))) {
    stop("`sens` must be a matrix or data.frame.")
  }
  if (nrow(A) != nrow(sens)) {
    stop("`A` and `sens` must have the same number of rows.")
  }
  if (!type %in% c("action.id", "leaf.id")) {
    stop("`type` must be either 'action.id' or 'leaf.id'.")
  }

  # Check that names(tree_list) match expected groups from sens
  group_keys <- apply(sens, 1, function(row) paste(row, collapse = "_"))
  required_keys <- unique(group_keys)
  missing_keys <- setdiff(required_keys, names(tree_list))
  if (length(missing_keys) > 0) {
    stop("`tree_list` is missing trees for the following sensitive attribute groups: ",
         paste(missing_keys, collapse = ", "))
  }
  ##-- Main prediction --##
  Dstar <- rep(NA_integer_, nrow(A))
  for(sens_unique in names(tree_list)){
    sens_values <- as.numeric(unlist(strsplit(sens_unique, "_")))
    if(length(sens_values) == 1) {
      bool <- sens[, 1] == sens_values
    } else {
      bool <- apply(sens, 1, function(row) all(row == sens_values))
    }
    filtered_A <- A[bool, , drop = FALSE]
    if (any(bool)) {
      Dstar[bool] <- predict(
        tree_list[[sens_unique]],
        filtered_A,
        type = type,
        seed = seed)
    }
  }
  return(Dstar)
}

#' Fit a Fair Probabilistic Split Tree
#'
#' This function performs a cdf-fairness adjustment on decision variables,
#' and optionally on policy score variables. It then fits a policy tree using the
#' `policytree` package, and adjusts split thresholds for each sensitive group
#' to produce probabilistic split trees.
#'
#' @param A A matrix or data.frame of decision variables.
#' @param scores A data.frame or matrix of policy score variables (one column per treatment option).
#' @param sens A data.frame or matrix of sensitive attributes used for fairness adjustment.
#' @param adjust_scores Logical. Whether to apply fairness adjustment to `scores` as well. Default is `FALSE`.
#' @param seed Integer seed for reproducibility. Default is 123456.
#' @param ties.method Character string for ranking ties. One of "random", "average", "first", "last", "max", "min".
#' @param depth Integer. Maximum depth of the output policy tree. Passed to `policytree::policy_tree`.
#' @param search.depth Integer. Only used if greater than `depth`. If so, hybrid tree search is applied
#'        using `policytree::hybrid_policy_tree`. Default is equal to `depth`.
#' @param split.step An optional approximation parameter, the number of possible splits to consider
#' when performing tree search. split.step = 1 (default) considers every possible
#' split, `split.step = 10` considers splitting at every 10'th sample and may yield
#' a substantial speedup for dense features. Manually rounding or re-encoding
#' continuous covariates with very high cardinality in a problem specific manner
#' allows for finer-grained control of the accuracy/runtime tradeoff and may in
#' some cases be the preferred approach..
#' @param min.node.size An integer indicating the smallest terminal node size permitted. Default is 1.
#' @param verbose Logical. Give verbose output. Default is `TRUE`.

#' @return A list of probabilistic split trees, one per sensitive group.
#'
#' @export
prob_split_tree <- function(
    A,
    scores,
    sens,
    adjust_scores=FALSE,
    seed = 123456,
    ties.method = "random",
    depth=2,
    search.depth = depth,
    split.step=1,
    min.node.size=1,
    verbose=TRUE){
  if (!is.logical(adjust_scores) || length(adjust_scores) != 1) {
    stop("`adjust_scores` must be a single logical value (TRUE or FALSE).")
  }
  As_cdf <- mq_adjustment(
   vars = A, sens = sens, seed = seed, ties.method = ties.method)$vars_cdf
  scores_use <- if (!adjust_scores) {
    scores
  }else{
    mq_adjustment(vars = scores, sens = sens, seed = seed, ties.method = ties.method)$vars_mq
  }
  tree_FUN <- if (depth == search.depth) {
    function(X, Gamma) {
      policytree::policy_tree(
        X = X,
        Gamma = Gamma,
        depth = depth,
        split.step = split.step,
        min.node.size = min.node.size,
        verbose = verbose
      )
    }
  } else {
    function(X, Gamma) {
      policytree::hybrid_policy_tree(
        X = X,
        Gamma = Gamma,
        depth = depth,
        search.depth = search.depth,
        split.step = split.step,
        min.node.size = min.node.size,
        verbose = verbose
      )
    }
  }
  opt_tree <- tree_FUN(X=As_cdf, Gamma=scores_use)
  # Transform the tree thresholds
  opt_tree_trans <- transform_tree(tree=opt_tree, sens=sens, vars=A, vars_cdf=As_cdf)
  return(opt_tree_trans)
}


#' Predict from a Probabilistic Split Tree
#'
#' @param tree A fitted object of class `'prob_split_tree'`.
#' @param vars A data.frame or matrix of the decision variables.
#' @param tree_cdf Optional. A corresponding `policy_tree` object fitted on CDF-adjusted decision variables.
#' @param vars_cdf Optional. A data.frame or matrix of CDF-adjusted decision variables (same structure as `vars`).
#' @param type Character. One of `"action.id"` (default) or `"leaf.id"`. Determines the type of prediction returned.
#' @param seed Integer. Random seed used to resolve probabilistic splits when exact info is missing. Default is 123456.
#'
#' @return A numeric vector of predicted action or leaf IDs for each observation.
#' @method predict prob_split_tree
#' @export
predict.prob_split_tree <- function(
    tree,
    vars,
    tree_cdf=NULL,
    vars_cdf=NULL,
    type="action.id", # or 'leaf.id'
    seed=123456){
  ##-- Input checks --##
  if (!inherits(tree, "prob_split_tree")) {
    stop("`tree` must be a fitted probabilistic split tree (class 'prob_split_tree').")
  }
  if (is.matrix(vars)) vars <- as.data.frame(vars)
  if (!is.data.frame(vars)) stop("`vars` must be a data.frame or matrix.")
  if (!is.null(tree_cdf) && !inherits(tree_cdf, "policy_tree")) {
    stop("If specified, `tree_cdf` must be of class 'policy_tree'.")
  }
  if (!is.null(vars_cdf)) {
    if (is.matrix(vars_cdf)) vars_cdf <- as.data.frame(vars_cdf)
    if (!is.data.frame(vars_cdf)) stop("`vars_cdf` must be a data.frame or matrix.")
    if (ncol(vars) != ncol(vars_cdf)) stop("`vars_cdf` must have same number of columns as `vars`.")
  }
  if (xor(is.null(tree_cdf), is.null(vars_cdf))) {
    message("To enable exact prediction, both `tree_cdf` and `vars_cdf` must be provided.")
  }
  if (!type %in% c("action.id", "leaf.id")) {
    stop("`type` must be either 'action.id' or 'leaf.id'.")
  }

  ##-- If tree is deterministic, fallback to policytree predict --##
  if (!tree$probabilistic_splits) {
    class(tree) <- "policy_tree"
    return(predict(tree, vars, type = type))
  }

  ##-- Probabilistic prediction path --##
  ta <- tree[["_tree_array"]]
  dft <- data.frame(matrix(FALSE, nrow=nrow(vars), ncol=nrow(ta)))
  colnames(dft) <- seq(nrow(ta))
  dft[, 1] <- TRUE
  if(!is.null(tree_cdf)) ta_fair <- tree_cdf[["_tree_array"]]
  # Loop over splits of original tree and adjust new trees
  for (row in seq_len(nrow(ta))){
    if (ta[row, 1] > 0){
      # Define splitting variable and threshold
      splitvar_trans <- tree[["columns"]][ta[row, 1]]
      splitvalue_trans <- ta[row, 2]
      left_child <- ta[row, 3]
      right_child <- ta[row, 4]
      # Get ids of current leaf and vars of splitting variable
      ids_current_leaf <- which(dft[, row])
      row.names(vars) <- NULL
      vars_current_leaf <- vars[ids_current_leaf, splitvar_trans]
      # Find ties
      dec <- ifelse(vars_current_leaf < splitvalue_trans, -1, ifelse(vars_current_leaf == splitvalue_trans, 0, 1))
      # Exact approach if fair tree provided
      if(!is.null(tree_cdf) & !is.null(vars_cdf)){
        splitvar_fair <- tree_cdf[["columns"]][ta[row, 1]]
        splitvalue_fair <- ta_fair[row, 2]
        row.names(vars_cdf) <- NULL
        vars_cdf_current_leaf <- vars_cdf[ids_current_leaf, splitvar_fair]
        # assign -1 and 1 accordingly to keep same result
        dec[dec == 0] <- (vars_cdf_current_leaf[dec == 0] > splitvalue_fair)*2-1
        # Probabilistic approach otherwise
      }else{
        # Retrieve share that goes left from tree object
        share_left <- ta[row, 5]
        # assign -1 and 1 according to share
        set.seed(seed)
        dec[dec == 0] <- (sample(seq(0, 1, length.out=sum(dec==0))) > share_left)*2-1
      }
      # Determine ids of left and right child
      go_left <- ids_current_leaf[which(dec <= 0)]
      go_right <- ids_current_leaf[which(dec > 0)]
      # Update dft
      dft[go_left, left_child] <- TRUE
      dft[go_right, right_child] <- TRUE
    }
    # Depending on requested type, return action.id or leaf.id
    final_nodes <- which(ta[, 1] < 0)
    node_id <- as.integer(colnames(dft[final_nodes])[max.col(dft[final_nodes])])
    if(type=='action.id'){
      result <- node_id*0
      all_ids <- unique(node_id)
      for(leaf in all_ids){
        result[node_id==leaf] <- ta[leaf, 2]
      }
    }else{
      result <- node_id
    }
  }
  return(result)
}


#' Plot a Probabilistic Split Tree.
#' @param tree A fitted object of class `'prob_split_tree'`.
#' @param leaf.labels An optional character vector of leaf labels for each treatment.

#' @method plot prob_split_tree
#' @export
plot.prob_split_tree <- function(tree, leaf.labels = NULL){
  # Check if prob_split_tree tree
  if (!inherits(tree, "prob_split_tree")) {
    stop("`tree` must be a fitted probabilistic split tree (class 'prob_split_tree').")
  }
  # Check if probabilistic split
  ##-- If tree is deterministic, fallback to policytree plot --##
  if (!tree$probabilistic_splits) {
    class(tree) <- "policy_tree"
    return(plot(tree, leaf.labels = leaf.labels))
  }
  # partly copied from plot.policy_tree
  if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
    stop("Package \"DiagrammeR\" must be installed to plot trees.")
  }
  n.actions <- tree$n.actions
  if (is.null(leaf.labels)) {
    leaf.labels <- paste("leaf node\n action =", 1:n.actions)
  } else if (length(leaf.labels) != n.actions) {
    stop("If provided, `leaf.labels` should be a vector with leaf labels for each treatment 1,..,K")
  }
  tree$leaf.labels <- leaf.labels
  # Get tree array of tree
  ta <- tree[["_tree_array"]]
  # Get number of splits
  n_splits <- sum(ta[,1] > 0)
  # Get plotting object
  dot_string <- policytree:::export_graphviz(tree)
  for (row in seq_len(nrow(ta))){
    if (ta[row, 1] > 0){
      # Create original string
      splitvar <- tree[["columns"]][ta[row, 1]]
      splitvalue <- ta[row, 2]
      share_left <- ta[row, 5]
      # Adjust split value if probabilistic split
      if(share_left < 1){
        org_string <- paste(splitvar, "<=", round(splitvalue, 2))
        new_string <- paste0(splitvar, " < ", round(splitvalue, 2), " (100%)\n", splitvar, " = ", round(splitvalue, 2), " (", round(share_left*100, 1),"%)")
        # Update dot string by replacing first occurence of org_string
        dot_string <- sub(org_string, new_string, dot_string)
      }
    }
  }
  DiagrammeR::grViz(dot_string)
}



#' Print a Probabilistic Split Tree.
#' @param tree A fitted object of class `'prob_split_tree'`.

#' @method print prob_split_tree
#' @export
print.prob_split_tree <- function(tree) {
  # Check if prob_split_tree tree
  if (!inherits(tree, "prob_split_tree")) {
    stop("`tree` must be a fitted probabilistic split tree (class 'prob_split_tree').")
  }
  # Check if probabilistic split
  ##-- If tree is deterministic, fallback to policytree print --##
  if (!tree$probabilistic_splits) {
    class(tree) <- "policy_tree"
    return(print(tree))
  }
  action.names <- if (all(tree$action.names == 1:tree$n.actions)) {
    1:tree$n.actions
  } else {
    paste0(1:tree$n.actions, ": ", tree$action.names)
  }
  cat("policy_tree object", "\n")
  cat("Tree depth: ", tree$depth, "\n")
  cat("Actions: ", action.names, "\n")
  cat("Variable splits:", "\n")

  # Add the index of each node as an attribute for easy access.
  nodes <- lapply(1:length(tree$nodes), function(i) {
    node <- tree$nodes[[i]]
    node$index <- i
    node$prob <- tree[['_tree_array']][i,5]
    return(node)
  })

  # Perform DFS to print the nodes (mimicking a stack with a list).
  frontier <- nodes[1]
  frontier[[1]]$depth <- 0
  while (length(frontier) > 0) {
    # Pop the first node off the stack.
    node <- frontier[[1]]
    frontier <- frontier[-1]

    output <- paste(rep("  ", node$depth), collapse = "")
    output <- paste(output, "(", node$index, ")", sep = "")

    if (node$is_leaf) {
      output <- paste(output, "* action:", node$action)
    } else {
      split.var <- node$split_variable
      split.var.name <- tree$columns[split.var]
      split.prob <- node$prob

      if(split.prob < 1){
        split.prob <- paste0(" (", round(node$prob*100, 2), "%)")
        output <- paste0(
          output,
          " split_variable: ",
          split.var.name,
          " split_value: < ",
          signif(node$split_value),
          " (100%) and = ",
          signif(node$split_value),
          split.prob)
      }else{
        output <- paste(output, "split_variable:", split.var.name, " split_value: <=", signif(node$split_value))
      }

      left_child <- nodes[node$left_child]
      left_child[[1]]$depth <- node$depth + 1

      right_child <- nodes[node$right_child]
      right_child[[1]]$depth <- node$depth + 1

      frontier <- c(left_child, right_child, frontier)
    }
    cat(output, "\n")
  }
}


#' Print a List of Probabilistic Split Trees.
#' @param tree A fitted object of class `'prob_split_tree_list'`.
#' @param sens_names The variables names of the sensitive attributes.
#'
#' @method print prob_split_tree_list
#' @export
print.prob_split_tree_list <- function(
    tree_list, sens_names=NULL){
  if (!inherits(tree_list, "prob_split_tree_list")) {
    stop("`tree_list` must be of class 'prob_split_tree_list'.")
  }
  cat("Probabilistic split trees:\n")
  for(sens_unique in names(tree_list)){
    sens_values <- as.numeric(unlist(strsplit(sens_unique, "_")))
    if (!is.null(sens_names)) {
      if (!is.character(sens_names)) {
        stop("`sens_names` must be a character vector.")
      }
      if (length(sens_names) != length(sens_values)) {
        stop("`sens_names` must have the same length as the number of sensitive attributes (",
             length(sens_values), ") in key: ", sens_unique)
      }
      sens_string <- paste(paste0(sens_names, ": ", sens_values), collapse = ", ")
    } else {
      sens_string <- sens_unique
    }
    cat(paste0("\nSenstive group: ", sens_string, "\n"))
    print(tree_list[[sens_unique]])
  }
}


#' Plot a List of Probabilistic Split Trees.
#' @param tree A fitted object of class `'prob_split_tree_list'`.
#' @param sens_names The variables names of the sensitive attributes.
#' @param leaf.labels An optional character vector of leaf labels for each treatment.
#'
#' @method plot prob_split_tree_list
#' @export
plot.prob_split_tree_list <- function(
    tree_list, sens_names = NULL, leaf.labels = NULL){
  if (!inherits(tree_list, "prob_split_tree_list")) {
    stop("`tree_list` must be of class 'prob_split_tree_list'.")
  }
  it <- 0
  plots <- list()
  long = ""
  for(sens_unique in names(tree_list)){
    it<-it+1
    sens_values <- as.numeric(unlist(strsplit(sens_unique, "_")))
    if (!is.null(sens_names)) {
      if (!is.character(sens_names)) {
        stop("`sens_names` must be a character vector.")
      }
      if (length(sens_names) != length(sens_values)) {
        stop("`sens_names` must have the same length as the number of sensitive attributes (",
             length(sens_values), ") in key: ", sens_unique)
      }
      sens_string <- paste(paste0(sens_names, ": ", sens_values), collapse = "\n")
    } else {
      sens_string <- sens_unique
    }
    plots[[sens_unique]] <- plot(tree_list[[sens_unique]], leaf.labels=leaf.labels)[["x"]][["diagram"]]
    plots[[sens_unique]] <- gsub("\\n(\\d+)", paste0("\n",as.character(it),"0\\1"), plots[[sens_unique]])
    plots[[sens_unique]] <- gsub("-> (\\d+)", paste0("-> ",as.character(it),"0\\1"), plots[[sens_unique]])
    plots[[sens_unique]] <- sub("digraph nodes { \n node [shape=box] ;\n", "", plots[[sens_unique]], fixed=TRUE)
    plots[[sens_unique]] <- sub("\n}", "", plots[[sens_unique]])
    plots[[sens_unique]] <- paste0(
      it, ' [color=none,style=filled, fillcolor=lightblue, label="', sens_string,
      '"];\n',it,' -> ',it,'00[style=invis];\n',plots[[sens_unique]])
    long <- paste0(long, plots[[sens_unique]])
  }
  DiagrammeR::grViz(paste0("digraph nodes { \n node [shape=box] ;\n", long, "\n}"))
}
