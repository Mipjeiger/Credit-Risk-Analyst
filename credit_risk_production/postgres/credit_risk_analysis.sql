SELECT * FROM credit_risk.data;

-- Portfolio overview
SELECT
    COUNT(*) AS total_applicants,
    COUNT(*) FILTER (WHERE approved_flag IS NOT NULL) AS applicants_with_decision,
    ROUND(AVG(age), 2) AS avg_age,
    ROUND(AVG(netmonthlyincome), 2) AS avg_monthly_income,
    ROUND(AVG(credit_score), 2) AS avg_credit_score,
    ROUND(AVG(tot_active_tl), 2) AS avg_active_accounts,
    ROUND(AVG(tot_missed_pmnt), 2) AS avg_missed_payments
FROM credit_risk.data;

-- Approval / risk-class distribution
SELECT
	approved_flag,
	COUNT(*) AS applicant_count,
	ROUND(
		100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
		2
 	) AS portofolio_percentage
FROM credit_risk.data
GROUP BY approved_flag
ORDER BY applicant_count DESC;

-- Approval class by average credit score
SELECT
	approved_flag,
	COUNT(*) AS applicants,
	ROUND(AVG(credit_score), 2) AS avg_credit_score,
	MIN(credit_score) AS min_credit_score,
	MAX(credit_score) AS max_credit_score
FROM credit_risk.data
GROUP BY approved_flag
ORDER BY avg_credit_score DESC;

-- Credit score risk segmentation
SELECT
	CASE
		WHEN credit_score < 600 THEN 'Very Low'
        WHEN credit_score < 650 THEN 'Low'
        WHEN credit_score < 700 THEN 'Medium'
        WHEN credit_score < 800 THEN 'High'
        ELSE 'Very High'
	END AS credit_score_band,
	COUNT(*) AS applicants,
	ROUND(AVG(netmonthlyincome), 2) AS avg_income,
	ROUND(AVG(tot_missed_pmnt), 2) AS avg_missed_payment,
	ROUND(AVG(tot_active_tl), 2) AS avg_active_accounts
FROM credit_risk.data
GROUP BY 1
ORDER BY MIN(credit_score);

-- Identify potentially high-risk applicants
SELECT
    prospectid,
    credit_score,
    netmonthlyincome,
    tot_active_tl,
    unsecured_tl,
    tot_missed_pmnt,
    num_times_delinquent,
    num_times_30p_dpd,
    num_times_60p_dpd,
    cc_utilization,
    pl_utilization,
    tot_enq,
    approved_flag
FROM credit_risk.data
WHERE
    (
        credit_score < 600
        OR tot_missed_pmnt >= 3
        OR num_times_60p_dpd >= 1
        OR num_times_delinquent >= 3
        OR cc_utilization >= 75
        OR pl_utilization >= 75
        OR tot_enq >= 10
    )
ORDER BY
    credit_score ASC,
    tot_missed_pmnt DESC,
    num_times_60p_dpd DESC;

-- INNER JOIN: Applicant + Credit Score Segment
WITH score_segment AS (
	SELECT
		prospectid,
		CASE
			WHEN credit_score < 600 THEN 'Very High Risk'
	        WHEN credit_score < 650 THEN 'High Risk'
	        WHEN credit_score < 700 THEN 'Medium Risk'
	        WHEN credit_score < 800 THEN 'Low Risk'
	        ELSE 'Very Low Risk'
		END AS score_segment
	FROM credit_risk.data
)

SELECT
	d.prospectid,
	d.credit_score,
	s.score_segment,
	d.approved_flag,
	d.netmonthlyincome
FROM credit_risk.data d
INNER JOIN score_segment s
	ON d.prospectid = s.prospectid
ORDER BY d.credit_score;

-- LEFT JOIN Applicant + Delinquency Classification
WITH delinquency_segment AS (
	SELECT
		prospectid,
		CASE
			WHEN num_times_delinquent = 0 THEN 'No Delinquency'
			WHEN num_times_delinquent <= 2 THEN 'LOW Delinquency'
			WHEN num_times_delinquent <= 5 THEN 'Moderate Delinquency'
			ELSE 'Severe Delinquency'
		END AS delinquency_segment
	FROM credit_risk.data
)

SELECT
	d.prospectid,
	d.credit_score,
	d.approved_flag,
	d.num_times_delinquent,
	s.delinquency_segment
FROM credit_risk.data d
LEFT JOIN delinquency_segment s
	ON d.prospectid = s.prospectid
ORDER BY d.num_times_delinquent DESC;

-- Identify approved flag (credit scoring risk)
SELECT
	approved_flag,
	COUNT(*) AS applicants,
	ROUND(
		100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
		2
	) AS percentage
FROM credit_risk.data
GROUP BY approved_flag
ORDER BY approved_flag;

-- Approval credit card by credit score credit
WITH score_credit AS (
	SELECT
		prospectid,
		approved_flag,
		credit_score,
		CASE
			WHEN credit_score < 600 THEN '<600'
			WHEN credit_score < 650 THEN '650-699'
			WHEN credit_score < 700 THEN '700-799'
			ELSE '800+'
		END AS score_credit
	FROM credit_risk.data
)
SELECT
	score_credit,
	COUNT(*) AS applicants,
	COUNT(*) FILTER (
		WHERE approved_flag = 'P1'
	) AS p1_auto_approved,

	COUNT(*) FILTER (
		WHERE approved_flag = 'P2'
	) AS p2_standard_approved,

	COUNT(3) FILTER (
		WHERE approved_flag = 'P3'
	) AS p3_conditional_approved,

	COUNT(3) FILTER (
		WHERE approved_flag = 'P4'
	) AS p4_direct_reject_approved,	

	ROUND(
		100.0 * COUNT(*) FILTER (WHERE approved_flag = 'P1')
		/ NULLIF(COUNT(*), 0),
		2
	) AS p1_rate
FROM score_credit
GROUP BY score_credit
ORDER BY
	CASE score_credit
		WHEN '<600' THEN 1
        WHEN '600-649' THEN 2
        WHEN '650-699' THEN 3
        WHEN '700-799' THEN 4
        WHEN '800+' THEN 5
    END;

-- High-Risk borrower identification
SELECT
	prospectid,
	credit_score,
	num_times_delinquent,
	num_times_60p_dpd,
	tot_missed_pmnt,
	cc_utilization,
	pl_utilization,
	tot_enq,
	approved_flag
FROM credit_risk.data
WHERE
		credit_score < 610
	OR num_times_delinquent >= 3
	OR num_times_60p_dpd > 0
	OR tot_missed_pmnt >= 3
    OR cc_utilization >= 75
    OR pl_utilization >= 75
ORDER BY credit_score ASC;

-- 30 + / 60+ DPD Risk Analysis
SELECT
	CASE
		WHEN num_times_60p_dpd > 0
            THEN '60+ DPD'

        WHEN num_times_30p_dpd > 0
            THEN '30+ DPD'

		ELSE 'No Serious DPD'
	END AS dpd_segment,

	COUNT(*) AS applicants,

	ROUND(AVG(credit_score), 2) AS avg_credit_score,
	SUM(tot_missed_pmnt) AS sum_missed_payments,
	ROUND(AVG(netmonthlyincome), 2) AS avg_income

FROM credit_risk.data
GROUP BY dpd_segment
ORDER BY applicants DESC;