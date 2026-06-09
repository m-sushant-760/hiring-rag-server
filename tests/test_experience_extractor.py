"""
Tests for src/utils/experience_extractor.py

Validates the timeline parsing, date range extraction, post-graduation classification,
and robustness to single-newline vs double-newline resume formatting.
"""

from datetime import date
import pytest
from src.utils.experience_extractor import (
    extract_experience,
    _extract_date_ranges,
    _layer2_timeline,
    _classify_timeline,
)


def test_extract_date_ranges_spans():
    """Verify that _extract_date_ranges returns matches sorted by span start index."""
    text = "Work: Jan 2020 – Dec 2022. Intern: 06/2019 – 08/2019."
    ranges = _extract_date_ranges(text)
    
    # Check that they are sorted by occurrence
    assert len(ranges) == 2
    assert ranges[0][0] == date(2020, 1, 1)  # Jan 2020
    assert ranges[0][1] == date(2022, 12, 1)  # Dec 2022
    assert ranges[1][0] == date(2019, 6, 1)  # 06/2019
    assert ranges[1][1] == date(2019, 8, 1)  # 08/2019


def test_akshay_kanbur_single_newline_experience():
    """
    Test case mirroring Akshay Kanbur's resume format.
    The jobs are separated by single newlines, and one is 2025-04 to Present,
    the other is 2024-07 to 2025-03.
    """
    experience_text = (
        "DevOps Engineer 1, HashedIn by Deloitte - Bangalore, India\n"
        "April 2025 – Present\n"
        "Client: The Walt Disney Company\n"
        "• Managed multi-service EKS clusters...\n"
        "Associate Software Engineer - Trainee, Xcel Corp - Bangalore, India\n"
        "July 2024 – March 2025\n"
        "• Automated AWS infrastructure provisioning..."
    )
    education_text = "Jain College of Engineering, Belagavi, B.E in CSE\nOct 2020 – May 2024"

    profile = extract_experience(
        full_text=experience_text + "\n" + education_text,
        experience_text=experience_text,
        education_text=education_text,
    )

    # 2024-07 to 2025-03: 8 months
    # 2025-04 to Present (June 2026): 14 months
    # Total professional: 22 months ~ 1.8 years
    assert profile.total_years is not None
    assert abs(profile.total_years - 1.8) < 0.15
    assert len(profile.timeline) == 2
    
    # Check that both jobs got titles and companies
    titles = [t["title"] for t in profile.timeline]
    assert "DevOps Engineer 1, HashedIn by Deloitte - Bangalore, India" in titles
    assert "Associate Software Engineer - Trainee, Xcel Corp - Bangalore, India" in titles


def test_pre_graduation_classification():
    """Verify that internships before the graduation date are marked as pre_graduation."""
    experience_text = (
        "Software Engineer, Acme Corp\n"
        "Jan 2025 – Present\n"
        "Intern, BigTech Corp\n"
        "Jan 2023 – June 2023"
    )
    education_text = "B.Tech in CS, Class of 2024"  # Grad year 2024 (cutoff June 2024)

    profile = extract_experience(
        full_text=experience_text + "\n" + education_text,
        experience_text=experience_text,
        education_text=education_text,
    )

    assert len(profile.timeline) == 2
    
    # Acme Corp is post-grad (professional)
    acme_job = [t for t in profile.timeline if "Acme" in t["title"] or "Acme" in t["company"] or "Acme" in (t.get("text") or t.get("title"))][0]
    assert acme_job["segment"] == "professional"
    
    # BigTech Corp is pre-grad (pre_graduation)
    bigtech_job = [t for t in profile.timeline if "BigTech" in t["title"] or "BigTech" in t["company"] or "BigTech" in (t.get("text") or t.get("title"))][0]
    assert bigtech_job["segment"] == "pre_graduation"
