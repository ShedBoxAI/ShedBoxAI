"""
Relationship detection between data sources for ShedBoxAI introspection.

This module provides sophisticated relationship detection capabilities that identify
foreign key relationships, column overlaps, and potential joins between data sources.
The detection algorithms are optimized for LLM consumption and provide actionable
recommendations for ShedBoxAI operations.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from .models import ColumnInfo, Relationship, SourceAnalysis


@dataclass
class RelationshipCandidate:
    """Internal class for relationship analysis"""

    source_a: str
    source_b: str
    field_a: str
    field_b: str
    match_type: str  # 'exact_name', 'similar_name', 'value_overlap', 'pk_fk_pattern'
    confidence_score: float
    evidence: List[str]
    sample_matches: List[Any] = None


class RelationshipDetector:
    """Detects relationships between data sources for LLM optimization"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration for relationship detection
        self.min_confidence_threshold = 0.5
        self.max_relationships_per_pair = 3
        self.sample_size_for_validation = 100

        # Common ID field patterns
        self.id_patterns = [
            r".*_id$",  # user_id, customer_id
            r"^id_.*",  # id_user, id_customer
            r"^id$",  # just 'id'
            r".*id$",  # userid, customerid (after lowercase)
            r".*key$",  # primary_key, foreign_key, primarykey
            r".*_key$",  # foreign_key, primary_key
        ]

        # Stop words for field name similarity
        self.stop_words = {"id", "key", "name", "type", "value", "data", "info"}

    def detect_relationships(self, analyses: Dict[str, SourceAnalysis]) -> List[Relationship]:
        """
        Detect relationships between all data sources.

        Args:
            analyses: Dictionary of source name to analysis results

        Returns:
            List of detected relationships sorted by confidence
        """
        self.logger.info(f"Detecting relationships between {len(analyses)} data sources")

        successful_analyses = {
            name: analysis
            for name, analysis in analyses.items()
            if analysis.success and analysis.schema_info and analysis.schema_info.columns
        }

        if len(successful_analyses) < 2:
            self.logger.info("Need at least 2 successful analyses to detect relationships")
            return []

        relationships = []
        source_pairs = self._get_source_pairs(list(successful_analyses.keys()))

        for source_a, source_b in source_pairs:
            analysis_a = successful_analyses[source_a]
            analysis_b = successful_analyses[source_b]

            pair_relationships = self._detect_pair_relationships(source_a, analysis_a, source_b, analysis_b)
            relationships.extend(pair_relationships)

        # Sort by confidence and limit results
        relationships.sort(key=lambda r: r.confidence, reverse=True)

        # Filter and validate top relationships
        validated_relationships = self._validate_relationships(relationships, successful_analyses)

        self.logger.info(f"Detected {len(validated_relationships)} high-confidence relationships")
        return validated_relationships

    def _get_source_pairs(self, source_names: List[str]) -> List[Tuple[str, str]]:
        """Generate all unique pairs of data sources"""
        pairs = []
        for i, source_a in enumerate(source_names):
            for source_b in source_names[i + 1 :]:
                pairs.append((source_a, source_b))
        return pairs

    def _detect_pair_relationships(
        self,
        source_a: str,
        analysis_a: SourceAnalysis,
        source_b: str,
        analysis_b: SourceAnalysis,
    ) -> List[Relationship]:
        """Detect relationships between a pair of data sources"""

        self.logger.debug(f"Analyzing relationship between {source_a} and {source_b}")

        columns_a = analysis_a.schema_info.columns
        columns_b = analysis_b.schema_info.columns

        if not columns_a or not columns_b:
            return []

        candidates = []

        # 1. Exact field name matches
        candidates.extend(self._find_exact_name_matches(source_a, columns_a, source_b, columns_b))

        # 2. Similar field name matches (fuzzy matching)
        candidates.extend(self._find_similar_name_matches(source_a, columns_a, source_b, columns_b))

        # 3. ID pattern matches
        candidates.extend(self._find_id_pattern_matches(source_a, columns_a, source_b, columns_b))

        # 4. Value overlap analysis (for string fields)
        candidates.extend(
            self._find_value_overlap_matches(source_a, columns_a, analysis_a, source_b, columns_b, analysis_b)
        )

        # Convert candidates to relationships
        relationships = self._candidates_to_relationships(candidates)

        # Limit relationships per pair
        return relationships[: self.max_relationships_per_pair]

    def _find_exact_name_matches(
        self,
        source_a: str,
        columns_a: List[ColumnInfo],
        source_b: str,
        columns_b: List[ColumnInfo],
    ) -> List[RelationshipCandidate]:
        """Find exact field name matches"""
        candidates = []

        names_a = {col.name.lower(): col for col in columns_a}
        names_b = {col.name.lower(): col for col in columns_b}

        common_names = set(names_a.keys()) & set(names_b.keys())

        for name in common_names:
            col_a = names_a[name]
            col_b = names_b[name]

            # Calculate confidence based on field characteristics
            confidence = self._calculate_exact_match_confidence(col_a, col_b)

            if confidence > self.min_confidence_threshold:
                candidates.append(
                    RelationshipCandidate(
                        source_a=source_a,
                        source_b=source_b,
                        field_a=col_a.name,
                        field_b=col_b.name,
                        match_type="exact_name",
                        confidence_score=confidence,
                        evidence=[
                            f"Exact field name match: '{col_a.name}'",
                            f"Type compatibility: {col_a.type} ↔ {col_b.type}",
                        ],
                    )
                )

        return candidates

    def _find_similar_name_matches(
        self,
        source_a: str,
        columns_a: List[ColumnInfo],
        source_b: str,
        columns_b: List[ColumnInfo],
    ) -> List[RelationshipCandidate]:
        """Find similar field name matches using fuzzy matching"""
        candidates = []

        for col_a in columns_a:
            for col_b in columns_b:
                similarity = self._calculate_name_similarity(col_a.name, col_b.name)

                if similarity > 0.7:  # High similarity threshold
                    confidence = similarity * self._calculate_type_compatibility(col_a, col_b)

                    if confidence > self.min_confidence_threshold:
                        candidates.append(
                            RelationshipCandidate(
                                source_a=source_a,
                                source_b=source_b,
                                field_a=col_a.name,
                                field_b=col_b.name,
                                match_type="similar_name",
                                confidence_score=confidence,
                                evidence=[
                                    f"Similar field names: '{col_a.name}' ≈ '{col_b.name}' ({similarity:.2f})",
                                    f"Type compatibility: {col_a.type} ↔ {col_b.type}",
                                ],
                            )
                        )

        return candidates

    def _find_id_pattern_matches(
        self,
        source_a: str,
        columns_a: List[ColumnInfo],
        source_b: str,
        columns_b: List[ColumnInfo],
    ) -> List[RelationshipCandidate]:
        """Find ID pattern matches (primary key to foreign key relationships)"""
        candidates = []

        # Find likely primary keys in source A
        pk_candidates_a = [col for col in columns_a if self._is_likely_primary_key(col)]

        # Find likely foreign keys in source B
        fk_candidates_b = [col for col in columns_b if self._is_likely_foreign_key(col)]

        # Cross-match PK to FK
        for pk_col in pk_candidates_a:
            for fk_col in fk_candidates_b:
                confidence = self._calculate_pk_fk_confidence(pk_col, fk_col, source_a, source_b)

                if confidence > self.min_confidence_threshold:
                    candidates.append(
                        RelationshipCandidate(
                            source_a=source_a,
                            source_b=source_b,
                            field_a=pk_col.name,
                            field_b=fk_col.name,
                            match_type="pk_fk_pattern",
                            confidence_score=confidence,
                            evidence=[
                                (
                                    f"Primary key pattern: {pk_col.name} "
                                    f"(unique: {pk_col.unique_count/max(pk_col.unique_count, 1):.2f})"
                                ),
                                f"Foreign key pattern: {fk_col.name}",
                                f"ID pattern match between {source_a} and {source_b}",
                            ],
                        )
                    )

        # Also check reverse direction (B to A)
        pk_candidates_b = [col for col in columns_b if self._is_likely_primary_key(col)]
        fk_candidates_a = [col for col in columns_a if self._is_likely_foreign_key(col)]

        for pk_col in pk_candidates_b:
            for fk_col in fk_candidates_a:
                confidence = self._calculate_pk_fk_confidence(pk_col, fk_col, source_b, source_a)

                if confidence > self.min_confidence_threshold:
                    candidates.append(
                        RelationshipCandidate(
                            source_a=source_a,
                            source_b=source_b,
                            field_a=fk_col.name,
                            field_b=pk_col.name,
                            match_type="fk_pk_pattern",
                            confidence_score=confidence,
                            evidence=[
                                f"Foreign key pattern: {fk_col.name}",
                                (
                                    f"Primary key pattern: {pk_col.name} "
                                    f"(unique: {pk_col.unique_count/max(pk_col.unique_count, 1):.2f})"
                                ),
                                f"ID pattern match between {source_a} and {source_b}",
                            ],
                        )
                    )

        return candidates

    def _find_value_overlap_matches(
        self,
        source_a: str,
        columns_a: List[ColumnInfo],
        analysis_a: SourceAnalysis,
        source_b: str,
        columns_b: List[ColumnInfo],
        analysis_b: SourceAnalysis,
    ) -> List[RelationshipCandidate]:
        """Find relationships based on value overlap analysis"""
        candidates = []

        # Only analyze if we have sample data
        if not analysis_a.sample_data or not analysis_b.sample_data:
            return candidates

        # Extract field values from sample data
        values_a = self._extract_field_values(analysis_a.sample_data)
        values_b = self._extract_field_values(analysis_b.sample_data)

        # Compare string fields for value overlap
        string_fields_a = [col for col in columns_a if col.type == "string"]
        string_fields_b = [col for col in columns_b if col.type == "string"]

        for col_a in string_fields_a:
            field_values_a = values_a.get(col_a.name, set())
            if not field_values_a:
                continue

            for col_b in string_fields_b:
                field_values_b = values_b.get(col_b.name, set())
                if not field_values_b:
                    continue

                # Calculate value overlap
                overlap_ratio = self._calculate_value_overlap(field_values_a, field_values_b)

                if overlap_ratio > 0.3:  # At least 30% overlap
                    confidence = overlap_ratio * 0.8  # Max confidence 0.8 for value overlap

                    if confidence > self.min_confidence_threshold:
                        common_values = field_values_a & field_values_b

                        candidates.append(
                            RelationshipCandidate(
                                source_a=source_a,
                                source_b=source_b,
                                field_a=col_a.name,
                                field_b=col_b.name,
                                match_type="value_overlap",
                                confidence_score=confidence,
                                evidence=[
                                    f"Value overlap: {overlap_ratio:.1%} ({len(common_values)} common values)",
                                    f"Sample common values: {list(common_values)[:3]}",
                                ],
                                sample_matches=list(common_values)[:10],
                            )
                        )

        return candidates

    def _calculate_exact_match_confidence(self, col_a: ColumnInfo, col_b: ColumnInfo) -> float:
        """Calculate confidence for exact name matches"""
        base_confidence = 0.9  # High base confidence for exact matches

        # Boost confidence for ID fields
        if self._is_likely_identifier_field(col_a.name):
            base_confidence = 0.95

        # Apply type compatibility modifier
        type_modifier = self._calculate_type_compatibility(col_a, col_b)

        # Apply uniqueness compatibility modifier
        uniqueness_modifier = self._calculate_uniqueness_compatibility(col_a, col_b)

        return base_confidence * type_modifier * uniqueness_modifier

    def _calculate_name_similarity(self, name_a: str, name_b: str) -> float:
        """Calculate similarity between field names using multiple methods"""
        name_a_clean = self._clean_field_name(name_a.lower())
        name_b_clean = self._clean_field_name(name_b.lower())

        # Exact match after cleaning
        if name_a_clean == name_b_clean:
            return 1.0

        # Levenshtein distance similarity
        levenshtein_sim = self._levenshtein_similarity(name_a_clean, name_b_clean)

        # Token-based similarity (for compound names like user_id vs customer_id)
        token_sim = self._token_similarity(name_a_clean, name_b_clean)

        # Longest common subsequence similarity
        lcs_sim = self._lcs_similarity(name_a_clean, name_b_clean)

        # Weighted combination
        return levenshtein_sim * 0.4 + token_sim * 0.4 + lcs_sim * 0.2

    def _clean_field_name(self, name: str) -> str:
        """Clean field name for comparison"""
        # Remove common prefixes/suffixes and special characters
        cleaned = re.sub(r"[_\-\s]+", "_", name.lower())

        # Remove common stop words at the end
        for stop_word in self.stop_words:
            if cleaned.endswith(f"_{stop_word}"):
                cleaned = cleaned[: -len(f"_{stop_word}")]
                break  # Only remove one stop word
            elif cleaned.startswith(f"{stop_word}_"):
                cleaned = cleaned[len(f"{stop_word}_") :]
                break  # Only remove one stop word

        # Handle direct matches (e.g., "userid" -> "user")
        if cleaned.endswith("id") and len(cleaned) > 2:
            cleaned = cleaned[:-2]
        elif cleaned.endswith("key") and len(cleaned) > 3:
            cleaned = cleaned[:-3]

        return cleaned

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein distance-based similarity"""
        if not s1 or not s2:
            return 0.0

        # Dynamic programming approach
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        max_len = max(m, n)
        return 1.0 - (dp[m][n] / max_len) if max_len > 0 else 0.0

    def _token_similarity(self, s1: str, s2: str) -> float:
        """Calculate token-based similarity"""
        if not s1 or not s2:
            return 0.0

        tokens1 = set(s1.split("_"))
        tokens2 = set(s2.split("_"))

        # Remove empty tokens
        tokens1 = {t for t in tokens1 if t}
        tokens2 = {t for t in tokens2 if t}

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union) if union else 0.0

    def _lcs_similarity(self, s1: str, s2: str) -> float:
        """Calculate longest common subsequence similarity"""
        if not s1 or not s2:
            return 0.0

        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        max_length = max(m, n)

        return lcs_length / max_length if max_length > 0 else 0.0

    def _calculate_type_compatibility(self, col_a: ColumnInfo, col_b: ColumnInfo) -> float:
        """Calculate type compatibility between columns"""
        type_a = col_a.type
        type_b = col_b.type

        # Exact type match
        if type_a == type_b:
            return 1.0

        # Compatible numeric types
        numeric_types = {"integer", "float", "numeric_string"}
        if type_a in numeric_types and type_b in numeric_types:
            return 0.9

        # String compatible types
        string_types = {"string", "date_string", "numeric_string"}
        if type_a in string_types and type_b in string_types:
            return 0.8

        # Date types
        date_types = {"date", "date_string"}
        if type_a in date_types and type_b in date_types:
            return 0.9

        # Boolean compatibility
        if (type_a == "boolean" and type_b in string_types) or (type_b == "boolean" and type_a in string_types):
            return 0.7

        # Default for incompatible types
        return 0.5

    def _calculate_uniqueness_compatibility(self, col_a: ColumnInfo, col_b: ColumnInfo) -> float:
        """Calculate uniqueness compatibility between columns"""
        # Use the uniqueness_ratio property from ColumnInfo
        ratio_a = col_a.uniqueness_ratio
        ratio_b = col_b.uniqueness_ratio

        # Both highly unique (likely identifiers)
        if ratio_a > 0.9 and ratio_b > 0.9:
            return 1.0

        # One unique, one not (PK-FK relationship)
        if (ratio_a > 0.9 and ratio_b < 0.8) or (ratio_b > 0.9 and ratio_a < 0.8):
            return 0.95

        # Both moderately unique
        if 0.5 < ratio_a < 0.9 and 0.5 < ratio_b < 0.9:
            return 0.8

        # Low uniqueness (categorical fields)
        if ratio_a < 0.3 and ratio_b < 0.3:
            return 0.7

        return 0.6

    def _is_likely_primary_key(self, col: ColumnInfo) -> bool:
        """Determine if column is likely a primary key"""
        return (
            col.uniqueness_ratio >= 0.95 and col.null_percentage == 0.0 and self._is_likely_identifier_field(col.name)
        )

    def _is_likely_foreign_key(self, col: ColumnInfo) -> bool:
        """Determine if column is likely a foreign key"""
        return (
            0.1 < col.uniqueness_ratio < 0.95  # Not unique but not too low
            and col.null_percentage < 10.0
            and self._is_likely_identifier_field(col.name)
        )

    def _is_likely_identifier_field(self, field_name: str) -> bool:
        """Check if field name matches ID patterns"""
        field_lower = field_name.lower()
        return any(re.match(pattern, field_lower) for pattern in self.id_patterns)

    def _calculate_pk_fk_confidence(
        self, pk_col: ColumnInfo, fk_col: ColumnInfo, pk_source: str, fk_source: str
    ) -> float:
        """Calculate confidence for primary key to foreign key relationships"""
        base_confidence = 0.7

        # Name similarity boost
        name_similarity = self._calculate_name_similarity(pk_col.name, fk_col.name)
        name_boost = name_similarity * 0.2

        # Type compatibility boost
        type_compatibility = self._calculate_type_compatibility(pk_col, fk_col)
        type_boost = max((type_compatibility - 0.5) * 0.2, 0)  # Only positive boost

        # PK characteristics boost
        pk_quality = pk_col.uniqueness_ratio
        pk_boost = max((pk_quality - 0.9) * 0.3, 0) if pk_quality > 0.9 else 0

        # FK characteristics boost (should not be unique)
        fk_quality = fk_col.uniqueness_ratio
        if fk_quality > 0.9:  # Too unique for FK, penalize
            fk_boost = -0.3  # Penalty for too unique FK
        else:
            fk_quality_adj = 1.0 - abs(fk_quality - 0.5)  # Optimal around 0.5
            fk_boost = fk_quality_adj * 0.1

        return min(base_confidence + name_boost + type_boost + pk_boost + fk_boost, 1.0)

    def _extract_field_values(self, sample_data: List[Dict[str, Any]]) -> Dict[str, Set[Any]]:
        """Extract field values from sample data for overlap analysis"""
        field_values = defaultdict(set)

        for record in sample_data:
            for field, value in record.items():
                if value is not None and isinstance(value, (str, int, float)):
                    # Convert to string for comparison
                    field_values[field].add(str(value))

        return dict(field_values)

    def _calculate_value_overlap(self, values_a: Set[Any], values_b: Set[Any]) -> float:
        """Calculate overlap ratio between two sets of values"""
        if not values_a or not values_b:
            return 0.0

        intersection = values_a & values_b
        union = values_a | values_b

        return len(intersection) / len(union) if union else 0.0

    def _candidates_to_relationships(self, candidates: List[RelationshipCandidate]) -> List[Relationship]:
        """Convert relationship candidates to final relationships"""
        relationships = []

        for candidate in candidates:
            # Determine relationship type
            if candidate.match_type in ["exact_name", "similar_name"]:
                rel_type = "field_match"
            elif candidate.match_type in ["pk_fk_pattern", "fk_pk_pattern"]:
                rel_type = "foreign_key"
            elif candidate.match_type == "value_overlap":
                rel_type = "value_overlap"
            else:
                rel_type = "potential_join"

            # Create description
            description = self._generate_relationship_description(candidate)

            # Calculate match percentage for value overlap
            match_percentage = None
            if candidate.sample_matches:
                # Rough estimate based on sample size
                match_percentage = min(len(candidate.sample_matches) * 10, 100)

            relationship = Relationship(
                source_a=candidate.source_a,
                source_b=candidate.source_b,
                type=rel_type,
                confidence=candidate.confidence_score,
                field_a=candidate.field_a,
                field_b=candidate.field_b,
                description=description,
                sample_matching_values=candidate.sample_matches or [],
                match_percentage=match_percentage,
            )

            relationships.append(relationship)

        return relationships

    def _generate_relationship_description(self, candidate: RelationshipCandidate) -> str:
        """Generate human-readable description for relationship"""
        confidence_desc = candidate.confidence_score

        if candidate.match_type == "exact_name":
            return f"Exact field name match with {confidence_desc:.0%} confidence"
        elif candidate.match_type == "similar_name":
            return f"Similar field names with {confidence_desc:.0%} confidence"
        elif candidate.match_type == "pk_fk_pattern":
            return f"Primary-foreign key relationship with {confidence_desc:.0%} confidence"
        elif candidate.match_type == "fk_pk_pattern":
            return f"Foreign-primary key relationship with {confidence_desc:.0%} confidence"
        elif candidate.match_type == "value_overlap":
            return f"Significant value overlap with {confidence_desc:.0%} confidence"
        else:
            return f"Potential data relationship with {confidence_desc:.0%} confidence"

    def _validate_relationships(
        self, relationships: List[Relationship], analyses: Dict[str, SourceAnalysis]
    ) -> List[Relationship]:
        """Validate and filter relationships"""
        validated = []

        # Track relationships per source pair to avoid duplicates
        pair_relationships = defaultdict(list)

        for rel in relationships:
            pair_key = tuple(sorted([rel.source_a, rel.source_b]))
            pair_relationships[pair_key].append(rel)

        # Keep only the best relationships per pair
        for pair_key, pair_rels in pair_relationships.items():
            # Sort by confidence and take top relationships
            pair_rels.sort(key=lambda r: r.confidence, reverse=True)

            # Add the best relationships, avoiding field duplicates
            used_fields = set()
            for rel in pair_rels:
                field_key = (rel.field_a, rel.field_b)
                reverse_field_key = (rel.field_b, rel.field_a)

                if field_key not in used_fields and reverse_field_key not in used_fields:
                    if rel.confidence >= self.min_confidence_threshold:
                        validated.append(rel)
                        used_fields.add(field_key)

                        # Limit relationships per pair
                        if (
                            len([r for r in validated if tuple(sorted([r.source_a, r.source_b])) == pair_key])
                            >= self.max_relationships_per_pair
                        ):
                            break

        return validated

    def generate_relationship_summary(self, relationships: List[Relationship]) -> Dict[str, Any]:
        """Generate summary statistics about detected relationships"""
        if not relationships:
            return {
                "total_relationships": 0,
                "by_type": {},
                "by_confidence": {},
                "recommendations": [],
            }

        # Count by type
        type_counts = defaultdict(int)
        for rel in relationships:
            type_counts[rel.type] += 1

        # Count by confidence level
        confidence_counts = {
            "very_high": len([r for r in relationships if r.confidence >= 0.9]),
            "high": len([r for r in relationships if 0.7 <= r.confidence < 0.9]),
            "medium": len([r for r in relationships if 0.5 <= r.confidence < 0.7]),
            "low": len([r for r in relationships if r.confidence < 0.5]),
        }

        # Generate recommendations
        recommendations = []

        if type_counts["foreign_key"] > 0:
            recommendations.append(
                "Foreign key relationships detected - consider using relationship_highlighting operations"
            )

        if type_counts["value_overlap"] > 0:
            recommendations.append("Value overlap detected - validate data consistency and consider data joins")

        high_confidence_rels = [r for r in relationships if r.confidence >= 0.8]
        if high_confidence_rels:
            recommendations.append(
                f"{len(high_confidence_rels)} high-confidence relationships found - suitable for automated joins"
            )

        return {
            "total_relationships": len(relationships),
            "by_type": dict(type_counts),
            "by_confidence": confidence_counts,
            "strongest_relationship": (max(relationships, key=lambda r: r.confidence) if relationships else None),
            "recommendations": recommendations,
        }
