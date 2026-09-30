# Запуск из корня этого репозитория. Пакеты: arrow, data.table, did, ggplot2, MatchIt.
library(arrow)
library(data.table)
library(did)
library(ggplot2)
library(MatchIt)

OUT <- "output"
if (!file.exists(file.path("input", "mcd_line_launches.csv"))) {
  stop("Запустите R из корня репозитория (папка с input/ и output/)")
}
if (!file.exists(file.path(OUT, "r_firm_year.parquet")) ||
    !file.exists(file.path(OUT, "r_hex_year.parquet"))) {
  stop("Сначала выполните notebook: нужны output/r_firm_year.parquet и output/r_hex_year.parquet")
}
SECTIONS <- c("G", "I", "J", "M", "Q", "S")
MIN_FIRMS_PER_ARM <- 30
ANTICIPATION <- 1  # год частичного запуска исключён из чистого pre-period
MATCH_G <- 2024     # 2020 = D1/D2; 2024 = D3/D4
firm_panel <- as.data.table(read_parquet(file.path(OUT, "r_firm_year.parquet")))
hexes <- as.data.table(read_parquet(file.path(OUT, "r_hex_year.parquet")))

firm_panel <- firm_panel[did_sample == 1 & stable_location_full == 1]
hexes <- hexes[did_sample == 1]
firm_panel[, firm_id_num := match(firm_id, unique(firm_id))]
hexes[, cell_id_num := match(cell_id, unique(cell_id))]

# ОКВЭД фиксируем по первой записи фирмы.
setorder(firm_panel, firm_id_num, year)
firm_panel[, baseline_section := {
  z <- okved_section[!is.na(okved_section)]
  if (length(z)) z[1] else NA_character_
}, by = firm_id_num]
firms <- firm_panel[did_preperiod_ok == 1] # прежняя строгая выборка для staggered DiD

# Числовой кластер: МЦД назначается целой линии, обычная ж/д — направлению.
all_corridors <- unique(c(firms$corridor_id, hexes$corridor_id))
firms[, cluster_num := match(corridor_id, all_corridors)]
hexes[, cluster_num := match(corridor_id, all_corridors)]
stopifnot(!anyNA(firms$cluster_num), !anyNA(hexes$cluster_num))

support <- unique(firms[, .(firm_id_num, baseline_section, g)])[
  , .(treated = sum(g > 0), ordinary_rail = sum(g == 0)),
  by = baseline_section
]
print(support[order(-treated)])
sections <- support[
  baseline_section %chin% SECTIONS &
    treated >= MIN_FIRMS_PER_ARM & ordinary_rail >= MIN_FIRMS_PER_ARM,
  baseline_section
]
if (!length(sections)) stop("Недостаточно treated/control фирм: проверьте реестр станций")

estimate_cs <- function(data, id, outcome, title) {
  data <- data[!is.na(get(outcome))]
  fit <- did::att_gt(
    yname = outcome, tname = "year", idname = id, gname = "g",
    data = as.data.frame(data), xformla = ~ 1,
    panel = TRUE, allow_unbalanced_panel = TRUE,
    control_group = "nevertreated", anticipation = ANTICIPATION,
    est_method = "dr", clustervars = "cluster_num",
    bstrap = TRUE, cband = TRUE, biters = 499
  )
  overall <- did::aggte(fit, type = "simple", na.rm = TRUE)
  event <- did::aggte(fit, type = "dynamic", min_e = -5, max_e = 4, na.rm = TRUE)
  plot <- did::ggdid(event) +
    labs(title = title, x = "Годы от первого полного года МЦД", y = "ATT") +
    theme_minimal(base_size = 12)
  list(fit = fit, overall = overall, event = event, plot = plot)
}

results <- list()
rows <- list()
for (section in sections) {
  for (level in c("firm", "hex")) {
    data <- if (level == "firm") firms[baseline_section == section] else hexes
    id <- if (level == "firm") "firm_id_num" else "cell_id_num"
    outcome <- if (level == "firm") "asinh_revenue_mln" else paste0("log1p_firms_", section)
    key <- paste(level, section, sep = "_")
    result <- tryCatch(
      estimate_cs(data, id, outcome, paste("МЦД:", level, "ОКВЭД", section)),
      error = function(e) {
        message("Пропущено ", key, ": ", conditionMessage(e))
        NULL
      }
    )
    if (is.null(result)) next
    results[[key]] <- result
    rows[[key]] <- data.table(
      level = level, section = section, outcome = outcome,
      ATT = result$overall$overall.att, SE = result$overall$overall.se
    )
    print(result$plot)
    ggsave(file.path(OUT, paste0("event_mcd_", key, ".png")),
           result$plot, width = 8, height = 5, dpi = 250)
  }
}
if (length(rows)) {
  staggered_table <- rbindlist(rows)
  print(staggered_table)
  fwrite(staggered_table, file.path(OUT, "mcd_staggered_att.csv"))
}

# Дополнение: matching на доступных pre-ковариатах выбранной когорты.
# Для каждого исхода отдельно оставляем фирмы с pre и post наблюдениями.
MATCH_PRE <- if (MATCH_G == 2020) 2016:2018 else 2020:2022
MATCH_POST <- if (MATCH_G == 2020) 2020:2022 else 2024:2025
# Для matching отдельная выборка: контроль не обязан наблюдаться с 2014 г.
# Доступность нужных pre/post-лет проверяется ниже отдельно для каждого исхода.
data_matching <- firm_panel[g %in% c(0L, MATCH_G)]
data_matching[, distance_center_km := sqrt(
  ((lon - 37.617698) * 111.32 * cos(55.755864 * pi / 180))^2 +
  ((lat - 55.755864) * 110.57)^2
)]
data_matching[, Treated := as.integer(g == MATCH_G)]

outcomes <- c(
  "Выручка" = "asinh_revenue_mln",
  "Издержки" = "asinh_costs_mln",
  "Прибыль" = "asinh_profit_mln"
)

match_one_outcome <- function(section, label, outcome) {
  panel <- data_matching[baseline_section == section]
  changes <- panel[, .(
    pre = mean(get(outcome)[year %in% MATCH_PRE], na.rm = TRUE),
    post = mean(get(outcome)[year %in% MATCH_POST], na.rm = TRUE),
    n_pre = sum(year %in% MATCH_PRE & !is.na(get(outcome))),
    n_post = sum(year %in% MATCH_POST & !is.na(get(outcome)))
  ), by = .(firm_id_num, Treated)]
  changes <- changes[n_pre > 0 & n_post > 0 & is.finite(pre) & is.finite(post)]
  changes[, change := post - pre]

  # Pre-ковариаты усредняем по доступным годам, без обязательного 2018/2022.
  base <- panel[year %in% MATCH_PRE, .(
    distance_center_km = mean(distance_center_km, na.rm = TRUE),
    lon = mean(lon, na.rm = TRUE),
    pre_revenue = mean(asinh_revenue_mln, na.rm = TRUE),
    pre_assets = mean(asinh_assets_mln, na.rm = TRUE)
  ), by = .(firm_id_num, Treated)]
  base <- merge(base, changes[, .(firm_id_num)], by = "firm_id_num")
  base <- base[complete.cases(base[, .(
    distance_center_km, lon, pre_revenue, pre_assets
  )])]
  n_treated <- base[, uniqueN(firm_id_num[Treated == 1])]
  n_control <- base[, uniqueN(firm_id_num[Treated == 0])]
  if (min(n_treated, n_control) < 20) return(NULL)

  matching <- MatchIt::matchit(
    Treated ~ distance_center_km + lon + pre_revenue + pre_assets,
    data = as.data.frame(base), method = "nearest", estimand = "ATT",
    ratio = 1, distance = "glm"
  )
  matched <- as.data.table(MatchIt::match.data(matching))[
    weights > 0, .(firm_id_num, subclass = as.character(subclass))
  ]
  if (!nrow(matched) || anyNA(matched$subclass)) return(NULL)
  changes <- merge(changes, matched, by = "firm_id_num")
  pairs <- dcast(changes, subclass ~ Treated, value.var = "change")
  if (!all(c("0", "1") %in% names(pairs))) return(NULL)
  pairs <- pairs[is.finite(get("0")) & is.finite(get("1"))]
  if (nrow(pairs) < 2L) return(NULL)
  pairs[, pair_effect := get("1") - get("0")]
  att <- mean(pairs$pair_effect)
  se <- sd(pairs$pair_effect) / sqrt(nrow(pairs))
  capture.output(
    summary(matching, standardize = TRUE),
    file = file.path(OUT, paste0("matching_balance_", section, "_", outcome, ".txt"))
  )
  data.table(
    section = section, outcome = label,
    ATT = att, SE = se, lo = att - 1.96 * se, hi = att + 1.96 * se,
    treated_firms = nrow(pairs), control_firms = nrow(pairs),
    matched_pairs = nrow(pairs), treated_with_outcome = n_treated,
    control_with_outcome = n_control,
    treated_retained_share = nrow(pairs) / n_treated
  )
}

designs <- CJ(section = sections, label = names(outcomes), unique = TRUE)
matched_rows <- lapply(seq_len(nrow(designs)), function(i) {
  section <- designs$section[i]
  label <- designs$label[i]
  tryCatch(match_one_outcome(section, label, outcomes[[label]]), error = function(e) {
    message("Matching пропущен для ", section, "/", label, ": ", conditionMessage(e))
    NULL
  })
})
matched_rows <- Filter(function(x) !is.null(x) && nrow(x) > 0, matched_rows)
if (length(matched_rows)) {
  matched_table <- rbindlist(matched_rows)
  print(matched_table)
  fwrite(matched_table, file.path(OUT, "mcd_matched_heterogeneity.csv"))
  heterogeneity_plot <- ggplot(matched_table, aes(x = ATT, y = reorder(section, ATT))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.2, orientation = "y") +
    geom_point(size = 2.5, color = "#2166ac") +
    facet_wrap(~ outcome, scales = "free_x") +
    labs(title = paste("МЦД: matched DiD, когорта", MATCH_G),
         x = "Разница изменений", y = "Секция ОКВЭД") +
    theme_minimal(base_size = 12)
  print(heterogeneity_plot)
  ggsave(file.path(OUT, paste(MATCH_G, "mcd_matched_heterogeneity.png")),
         heterogeneity_plot, width = 10, height = 6, dpi = 250)
}

# Гексы: число фирм (все и по секциям) + общие денежные показатели.
# Пустой гекс = 0; если фирмы есть, но никто не отчитался, деньги = NA.
hexes[, revenue_for_match := fifelse(
  active_firms == 0, 0, asinh(total_revenue / 1e6)
)]
hexes[, assets_for_match := fifelse(
  active_firms == 0, 0, asinh(total_assets / 1e6)
)]
hex_outcomes <- c(
  firm_count_all = "log1p_active_firms",
  revenue_total = "revenue_for_match",
  assets_total = "assets_for_match",
  setNames(paste0("log1p_firms_", SECTIONS), paste0("firm_count_", SECTIONS))
)
hex_labels <- c(
  firm_count_all = "Все фирмы",
  revenue_total = "Общая выручка",
  assets_total = "Общие активы",
  setNames(paste0("Фирмы ОКВЭД ", SECTIONS), paste0("firm_count_", SECTIONS))
)
hex_matching <- hexes[g %in% c(0L, MATCH_G)]
hex_matching[, Treated := as.integer(g == MATCH_G)]

match_hex_outcome <- function(key, outcome) {
  changes <- hex_matching[, .(
    pre = mean(get(outcome)[year %in% MATCH_PRE], na.rm = TRUE),
    post = mean(get(outcome)[year %in% MATCH_POST], na.rm = TRUE),
    n_pre = sum(year %in% MATCH_PRE & !is.na(get(outcome))),
    n_post = sum(year %in% MATCH_POST & !is.na(get(outcome)))
  ), by = .(cell_id_num, Treated)]
  changes <- changes[n_pre > 0 & n_post > 0 & is.finite(pre) & is.finite(post)]
  changes[, change := post - pre]

  base <- hex_matching[year %in% MATCH_PRE, .(
    distance_m = mean(distance_m, na.rm = TRUE),
    pre_outcome = mean(get(outcome), na.rm = TRUE),
    pre_total_firms = mean(log1p_active_firms, na.rm = TRUE)
  ), by = .(cell_id_num, Treated)]
  base <- merge(base, changes[, .(cell_id_num)], by = "cell_id_num")
  base <- base[complete.cases(base[, .(distance_m, pre_outcome, pre_total_firms)])]
  n_treated <- base[, uniqueN(cell_id_num[Treated == 1])]
  n_control <- base[, uniqueN(cell_id_num[Treated == 0])]
  if (min(n_treated, n_control) < 20) return(NULL)

  formula <- if (key == "firm_count_all") {
    Treated ~ distance_m + pre_outcome
  } else {
    Treated ~ distance_m + pre_outcome + pre_total_firms
  }
  matching <- MatchIt::matchit(
    formula, data = as.data.frame(base), method = "nearest",
    estimand = "ATT", ratio = 1, distance = "glm"
  )
  matched <- as.data.table(MatchIt::match.data(matching))[
    weights > 0, .(cell_id_num, subclass = as.character(subclass))
  ]
  if (!nrow(matched) || anyNA(matched$subclass)) return(NULL)
  changes <- merge(changes, matched, by = "cell_id_num")
  pairs <- dcast(changes, subclass ~ Treated, value.var = "change")
  if (!all(c("0", "1") %in% names(pairs))) return(NULL)
  pairs <- pairs[is.finite(get("0")) & is.finite(get("1"))]
  if (nrow(pairs) < 2L) return(NULL)
  pairs[, pair_effect := get("1") - get("0")]
  att <- mean(pairs$pair_effect)
  se <- sd(pairs$pair_effect) / sqrt(nrow(pairs))
  capture.output(
    summary(matching, standardize = TRUE),
    file = file.path(OUT, paste0("matching_balance_hex_", key, ".txt"))
  )
  data.table(
    outcome_code = key, outcome = hex_labels[[key]], variable = outcome,
    ATT = att, SE = se, lo = att - 1.96 * se, hi = att + 1.96 * se,
    matched_pairs = nrow(pairs), treated_with_outcome = n_treated,
    control_with_outcome = n_control,
    treated_retained_share = nrow(pairs) / n_treated
  )
}

matched_hex_rows <- lapply(names(hex_outcomes), function(key) {
  tryCatch(match_hex_outcome(key, hex_outcomes[[key]]), error = function(e) {
    message("Гекс-matching пропущен для ", key, ": ", conditionMessage(e))
    NULL
  })
})
matched_hex_rows <- Filter(function(x) !is.null(x) && nrow(x) > 0, matched_hex_rows)
if (length(matched_hex_rows)) {
  matched_hex_table <- rbindlist(matched_hex_rows)
  print(matched_hex_table)
  fwrite(matched_hex_table, file.path(OUT, "mcd_matched_hex.csv"))
  matched_hex_table[, group := ifelse(
    outcome_code %chin% c("revenue_total", "assets_total"), "Деньги", "Число фирм"
  )]
  hex_plot <- ggplot(matched_hex_table%>%filter(group == 'Число фирм'), aes(x = ATT, y = reorder(outcome, ATT))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.2, orientation = "y") +
    geom_point(size = 2.5, color = "#2166ac") +
    facet_wrap(~ group, scales = "free_x") +
    labs(title = paste("МЦД: matched DiD на гексах, когорта", MATCH_G),
         x = "Разница изменений", y = NULL) +
    theme_minimal(base_size = 12)
  print(hex_plot)
  ggsave(file.path(OUT, paste(MATCH_G, "mcd_matched_hex.png")),
         hex_plot, width = 10, height = 6, dpi = 250)
}
