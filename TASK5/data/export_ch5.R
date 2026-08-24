# 导出第5章讲座格式数据：pg15training -> pg15training_ch5.csv
load("pg15training.rda")
df <- pg15training
cat("原始行数:", nrow(df), "\n")

# 讲座：移除前21条（理赔次数非零但理赔金额为零的重复观测）
dup <- which(df$Numtppd > 0 & df$Indtppd == 0)
cat("Numtppd>0 且 Indtppd==0 的行:", length(dup), "\n")
df2 <- df[!(seq_len(nrow(df)) %in% head(dup, 21)), ]
cat("移除后行数:", nrow(df2), "\n")
cat("CalYear 分布:\n")
print(table(df2$CalYear))

# 派生变量（按讲座：Exposure=风险敞口比例；ClaimFrequency=年化频率；PurePremium=年化纯保费）
df2$Exposure <- df2$Exppdays / 365
df2$ClaimNb <- df2$Numtppd
df2$ClaimTotal <- df2$Indtppd
df2$ClaimFrequency <- df2$Numtppd / df2$Exppdays * 365
df2$PurePremium <- df2$Indtppd / df2$Exppdays * 365

cat("Exposure range:", range(df2$Exposure), " mean:", mean(df2$Exposure), "\n")
cat("ClaimNb max:", max(df2$ClaimNb), " mean:", mean(df2$ClaimNb), "\n")
cat("ClaimTotal max:", max(df2$ClaimTotal), " mean:", mean(df2$ClaimTotal), "\n")
cat("ClaimFrequency max:", max(df2$ClaimFrequency), " mean:", mean(df2$ClaimFrequency), "\n")
cat("PurePremium max:", max(df2$PurePremium), " mean:", mean(df2$PurePremium), "\n")
cat("Age range:", range(df2$Age), " Bonus range:", range(df2$Bonus), "\n")

write.csv(df2, "pg15training_ch5.csv", row.names = FALSE)
cat("CSV 已导出, 列数:", ncol(df2), "\n")
