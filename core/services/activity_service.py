import anyio

import logging
import time

from core.db.connection import get_db_connection
from core.constants import ONTOLOGY_LABELS

logger = logging.getLogger(__name__)


async def get_activity_portfolio(codes: list[str], code_registry: list[dict]) -> dict:
    """
    Retrieves and aggregates grant data for the specified activity codes,
    including historical funding trends, ontology distribution, and detailed grant information.

    Parameters
    ----------
    codes : list[str]
        A list of activity codes to filter the grants by. Must contain between 1 and 5 codes.
    code_registry : list[dict]
        A list of dictionaries containing valid activity codes and their associated metadata.

    Returns
    -------
    dict
        A dictionary containing the aggregated results, including:
        - ``query``: a string representation of the activity codes.
        - ``years``: a list of fiscal years for which data is available.
        - ``funding``: a list of total funding amounts corresponding to each fiscal year.
        - ``results``: a list of detailed grant records for the specified activity codes.
        - ``ontology_labels``: a list of ontology category labels.
        - ``ontology_values``: a list of total funding amounts corresponding to each ontology category.
    """

    start = time.perf_counter()

    target_codes = [c.strip().upper() for c in codes if c.strip()]
    if not target_codes or len(target_codes) > 5:
        raise ValueError("Must provide between 1 and 5 activity codes.")

    logger.info("Retrieving portfolio for activity codes: %s", ", ".join(target_codes))

    tuple_codes = tuple(target_codes)

    conn = await anyio.to_thread.run_sync(get_db_connection)
    cur = conn.cursor()

    def run_database_operations():

        try:

            # STEP 1: HISTORICAL DATA FOR CHARTS

            # Strictly check against the 3-character activity code substring
            timeline_query = """
                SELECT fiscal_year, SUM(total_award_amount)
                FROM ResearchGrants
                WHERE activity_code IN %s
                GROUP BY fiscal_year
                ORDER BY fiscal_year ASC;
            """
            cur.execute(timeline_query, (tuple_codes,))
            timeline_rows = cur.fetchall()

            years = [row[0] for row in timeline_rows]
            funding = [float(row[1]) if row[1] else 0.0 for row in timeline_rows]

            # STEP 2: CATEGORY DISTRIBUTION AGGREGATION
            ontology_query = """
                SELECT               
                    COALESCE(SUM(CASE WHEN gl.mechanistic = 1 THEN rg.total_award_amount ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN gl.therapeutic = 1 THEN rg.total_award_amount ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN gl.diagnostic = 1 THEN rg.total_award_amount ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN gl.research_tool = 1 THEN rg.total_award_amount ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN gl.clinical = 1 THEN rg.total_award_amount ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN gl.infrastructure = 1 THEN rg.total_award_amount ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN gl.education = 1 THEN rg.total_award_amount ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN gl.obs_ep = 1 THEN rg.total_award_amount ELSE 0 END), 0)
                FROM ResearchGrants rg
                INNER JOIN grant_labels gl ON rg.grant_id = gl.grant_id
                WHERE rg.activity_code IN %s;
            """
            cur.execute(ontology_query, (tuple_codes,))
            onto_row = cur.fetchone()

            ontology_values = (
                [float(val) for val in onto_row] if onto_row else [0.0] * 8
            )

            # STEP 3: TABLE REVIEWS (2025 Viewport Only)
            table_query = """
                SELECT
                    rg.grant_id,
                    rg.project_title,
                    o.name AS organization,
                    rg.contact_pi_id,
                    rg.total_award_amount,
                    rg.agency_ic,
                    rg.fiscal_year,
                    gl.mechanistic, gl.therapeutic, gl.diagnostic, gl.research_tool,
                    gl.clinical, gl.infrastructure, gl.education, gl.obs_ep,               
                    s.two_sentence_summary AS summary_text
                FROM ResearchGrants rg
                JOIN grant_labels gl ON rg.grant_id = gl.grant_id
                INNER JOIN Grant_Summaries s ON rg.grant_id = s.grant_id
                LEFT JOIN organizations o ON rg.organization_id = o.id
                WHERE 
                    rg.activity_code IN %s
                    AND rg.fiscal_year = 2025
                ORDER BY rg.total_award_amount DESC;
            """
            cur.execute(table_query, (tuple_codes,))
            rows = cur.fetchall()

            grants = []
            for row in rows:
                grant_id = row[0]
                activity_code = grant_id[1:4] if len(grant_id) > 4 else "N/A"

                grants.append(
                    {
                        "grant_id": grant_id,
                        "title": row[1],
                        "organization": row[2] if row[2] else "Unknown Institution",
                        "pi": row[3],
                        "amount": float(row[4]) if row[4] else 0.0,
                        "agency_ic": row[5],
                        "fiscal_year": int(row[6]) if row[6] else 2025,
                        "mechanistic": row[7],
                        "therapeutic": row[8],
                        "diagnostic": row[9],
                        "research_tool": row[10],
                        "clinical": row[11],
                        "infrastructure": row[12],
                        "education": row[13],
                        "obs_ep": row[14],
                        "summary": row[15],
                        "activity_code": activity_code,
                    }
                )

            return years, funding, ontology_values, grants

        finally:
            cur.close()
            conn.close()

    try:
        years, funding, ontology_values, grants = await anyio.to_thread.run_sync(
            run_database_operations
        )

    except Exception as e:
        logger.exception(
            "Failed to retrieve activity portfolio for codes: %s",
            ", ".join(target_codes),
        )
        raise

    display_title = f"Activity Codes: {', '.join(target_codes)}"

    logger.info(
        "Retrieved %d grants for activity codes: %s in %.2f seconds",
        len(grants),
        ", ".join(target_codes),
        time.perf_counter() - start,
    )

    return {
        "query": display_title,
        "years": years,
        "funding": funding,
        "results": grants,
        "ontology_labels": ONTOLOGY_LABELS,
        "ontology_values": ontology_values,
    }
