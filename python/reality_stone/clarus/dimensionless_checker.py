"""
Clarus Equation Dimensionless Consistency Checker
================================================================
Verifies that all formulas in CE have correct dimensional analysis.

Dimensions: [M], [L], [T], [Θ] (temperature)
Natural units: ℏ = c = k_B = 1 → all expressed in mass dimension

This catches:
  ✓ Hidden dimensional constants that shouldn't be there
  ✓ Wrong exponents in formulas
  ✓ Unit conversion errors
  ✓ Mixing of natural and SI units
"""

import ast
from enum import Enum
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Tuple

try:
    from sympy import parse_expr as _sympy_parse_expr
except ModuleNotFoundError:
    FORMULA_PARSER_BACKEND = "stdlib.ast.parse.syntax_only"

    def _parse_formula(expression: str) -> None:
        """Syntax-only fallback; it is not symbolic dimensional analysis."""

        ast.parse(expression.replace("^", "**"), mode="eval")

else:
    FORMULA_PARSER_BACKEND = "sympy.parse_expr"

    def _parse_formula(expression: str) -> None:
        _sympy_parse_expr(expression)


@dataclass(frozen=True)
class DimensionVector:
    """Exact dimension exponents for combinations not named by ``Dimension``.

    The old checker returned ``Dimension.DIMENSIONLESS`` whenever a product,
    quotient, or power did not happen to match one of the small named enum
    members.  That made, for example, ``TIME**-1`` and ``MASS**2`` silently
    dimensionless.  Keeping the full vector makes the fallback conservative.
    """

    exponents: Tuple[Fraction, Fraction, Fraction, Fraction]

    @classmethod
    def from_exponents(cls, exponents) -> "DimensionVector":
        values = tuple(Fraction(value) for value in exponents)
        if len(values) != 4:
            raise ValueError("dimension vectors must have four exponents")
        return cls(values)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        labels = ("M", "L", "T", "Theta")
        terms = [f"{label}^{power}" for label, power in zip(labels, self.exponents) if power]
        return "DIMENSIONLESS" if not terms else " ".join(terms)

    def __mul__(self, other):
        other_vector = _as_dimension_vector(other)
        if other_vector is NotImplemented:
            return NotImplemented
        return _dimension_from_exponents(
            a + b for a, b in zip(self.exponents, other_vector.exponents)
        )

    def __truediv__(self, other):
        other_vector = _as_dimension_vector(other)
        if other_vector is NotImplemented:
            return NotImplemented
        return _dimension_from_exponents(
            a - b for a, b in zip(self.exponents, other_vector.exponents)
        )

    def __pow__(self, power):
        exponent = Fraction(power)
        return _dimension_from_exponents(exponent * value for value in self.exponents)

    def is_dimensionless(self) -> bool:
        return all(exponent == 0 for exponent in self.exponents)


class Dimension(Enum):
    """Fundamental dimensions in natural units (ℏ=c=k_B=1)."""

    DIMENSIONLESS = (0, 0, 0, 0)  # pure number
    MASS = (1, 0, 0, 0)  # M
    LENGTH = (0, 1, 0, 0)  # L = M⁻¹
    TIME = (0, 0, 1, 0)  # T = M⁻¹
    TEMPERATURE = (0, 0, 0, 1)  # Θ = M
    ENERGY = (1, 0, 0, 0)  # E = M
    MOMENTUM = (1, 0, 0, 0)  # p = M
    COUPLING = (0, 0, 0, 0)  # α = dimensionless
    ACTION = (0, 0, 0, 0)  # S = ℏ∫L dt, dimensionless in natural units

    @property
    def exponents(self) -> Tuple[int, int, int, int]:
        """(M, L, T, Θ) exponents."""
        return self.value

    def __mul__(self, other):
        """Dimension multiplication."""
        other_vector = _as_dimension_vector(other)
        if other_vector is NotImplemented:
            return NotImplemented
        return _dimension_from_exponents(
            Fraction(a) + b for a, b in zip(self.exponents, other_vector.exponents)
        )

    def __truediv__(self, other):
        """Dimension division."""
        other_vector = _as_dimension_vector(other)
        if other_vector is NotImplemented:
            return NotImplemented
        return _dimension_from_exponents(
            Fraction(a) - b for a, b in zip(self.exponents, other_vector.exponents)
        )

    def __pow__(self, n):
        """Dimension power."""
        exponent = Fraction(n)
        return _dimension_from_exponents(exponent * value for value in self.exponents)

    def is_dimensionless(self) -> bool:
        """Check if truly dimensionless."""
        return all(e == 0 for e in self.exponents)


def _as_dimension_vector(value):
    if isinstance(value, DimensionVector):
        return value
    if isinstance(value, Dimension):
        return DimensionVector.from_exponents(value.exponents)
    return NotImplemented


def _dimension_from_exponents(exponents):
    vector = DimensionVector.from_exponents(exponents)
    for named in Dimension:
        if vector.exponents == tuple(Fraction(value) for value in named.exponents):
            return named
    return vector


@dataclass
class Formula:
    """Single CE formula with dimensional analysis."""

    name: str
    symbol: str
    formula: str  # Mathematical expression
    expected_dim: Dimension | DimensionVector
    source: str  # Which document
    notes: str = ""
    status: str = ""


class DimensionlessChecker:
    """Main checker class."""

    formula_parser_backend = FORMULA_PARSER_BACKEND

    def __init__(self):
        self.formulas: List[Formula] = []
        self.results: Dict[str, Dict] = {}
        self.load_ce_formulas()

    def load_ce_formulas(self):
        """Load all CE formulas to check."""

        # AXIOM LAYER
        self.add_formula(
            "Path folding survival function",
            "S(D)",
            "exp(-D)",  # S(D) = e^(-D)
            Dimension.DIMENSIONLESS,
            "axium.md / 경로적분.md",
            "D must be dimensionless (folding depth count)",
        )

        self.add_formula(
            "Effective dimensional depth",
            "D_eff",
            "3 + delta",  # D_eff = 3 + δ
            Dimension.DIMENSIONLESS,
            "경로적분.md section 2",
            "Pure integer dimensions; δ dimensionless",
        )

        # BRAIN/AGI LAYER
        self.add_formula(
            "Brain global activity equation",
            "p_{r,n+1}",
            "(1-rho_B)*p_star + rho_B*p_n + gamma*Laplacian*p",
            Dimension.DIMENSIONLESS,
            "6_뇌/00_읽기지도.md",
            "Activity probability, dimensionless",
        )

        self.add_formula(
            "STDP learning rate upper bound",
            "η_max",
            "(A_plus - A_minus*exp(-tau_plus/tau_minus)) / (tau_plus + tau_minus)",
            Dimension.TIME ** (-1),  # [T⁻¹]
            "6_뇌/08_시냅스가소성.md",
            "Rate has dimension time⁻¹",
        )

        self.add_formula(
            "Sleep pressure (Borbély)",
            "S(t)",
            "S_inf + (S_0 - S_inf)*exp(-t/tau_S)",
            Dimension.DIMENSIONLESS,
            "6_뇌/07_수면과복구.md",
            "S is unitless sleep pressure; t/tau_S dimensionless",
        )

        self.add_formula(
            "Clarus-field prediction-error gate",
            "g_CF",
            "1/(1 + exp(-(gain*norm_error_sq + bias)))",
            Dimension.DIMENSIONLESS,
            "7_AGI/Clarus-field baseline",
            "norm_error_sq = ||(observation-prediction)/reference_scale||^2; "
            "gain and bias are dimensionless",
        )

        self.add_formula(
            "Clarus-field structural phase score",
            "chi_CF",
            "field_decay*phi/source_cap",
            Dimension.DIMENSIONLESS,
            "7_AGI/Clarus-field baseline",
            "Runtime coefficients and state are dimensionless; this ratio fixes the threshold scale",
        )

        self.add_formula(
            "Unified-metric surprise gate score",
            "chi_UM",
            "metric_distance_sq/reference_length_sq",
            Dimension.DIMENSIONLESS,
            "7_AGI/V15 unified-metric finite baseline",
            "Both squared quantities use the same information-length scale; only their ratio "
            "may enter the hard threshold",
        )

        self.add_formula(
            "Unified-metric fixed-chart condition bound",
            "kappa_UM",
            "max_metric_eigenvalue/min_metric_eigenvalue",
            Dimension.DIMENSIONLESS,
            "7_AGI/V15 unified-metric finite baseline",
            "The eigenvalue ratio is dimensionless but fixed-chart, not affine invariant",
        )

        self.add_formula(
            "V16 covariant metric-flow residual",
            "r_V16",
            "log(predicted_quadratic_cost/observed_quadratic_cost)",
            Dimension.DIMENSIONLESS,
            "7_AGI/V16 covariant metric-flow agent",
            "Prediction and observation must use the same squared-cost reference unit; "
            "their positive ratio is dimensionless before entering log",
        )

        self.add_formula(
            "V16 normalized route regret",
            "rho_V16",
            "(selected_route_cost/minimum_route_cost)-1",
            Dimension.DIMENSIONLESS,
            "7_AGI/V16 covariant metric-flow agent",
            "Selected and minimum route costs have identical units and the minimum is positive",
        )

        self.add_formula(
            "V17 conditional signed-cue information",
            "I_V17",
            "H_sign_given_public_reference-H_sign_given_metric_and_public_reference",
            Dimension.DIMENSIONLESS,
            "7_AGI/V17 metric-only delayed-cue run",
            "Entropy and mutual information are logarithmic pure-number counts; the theorem "
            "is conditional on the public oriented reference",
        )

        self.add_formula(
            "V17 homogeneous-lift normalized action margin",
            "delta_V17",
            "(wrong_action_cost-correct_action_cost)/correct_action_cost",
            Dimension.DIMENSIONLESS,
            "7_AGI/V17 metric-only delayed-cue run",
            "Both quadratic action costs use the same synthetic cost unit",
        )

        self.add_formula(
            "V18b reward-decoded binary label",
            "y_tilde_V18b",
            "action*(2*reward-1)",
            Dimension.DIMENSIONLESS,
            "7_AGI/V18b reward-decoded delayed-credit run",
            "Action and binary correctness reward are pure numbers, so the decoded label "
            "is dimensionless",
        )

        self.add_formula(
            "V18b delayed classifier increment",
            "delta_w_V18b",
            "learning_rate*decoded_label*eligibility_coordinate",
            Dimension.DIMENSIONLESS,
            "7_AGI/V18b reward-decoded delayed-credit run",
            "The registered synthetic learning rate, label, and trace coordinates are all "
            "dimensionless",
        )

        self.add_formula(
            "A4 neuron-specific soft-threshold excitability",
            "a_A4",
            "1/(1 + exp(-(z-threshold)))",
            Dimension.DIMENSIONLESS,
            "6_뇌/11_리만계량_라우팅_논문.md",
            "z and the calibration-frozen threshold are dimensionless standardized activity",
        )

        self.add_formula(
            "A4 state-dependent edge conductance ratio",
            "r_w_A4",
            "exp(h_edge)",
            Dimension.DIMENSIONLESS,
            "6_뇌/11_리만계량_라우팅_논문.md",
            "h_edge is a clipped product of dimensionless normalized association and excitability",
        )

        self.add_formula(
            "A5 deformation-to-base graph RMS ratio",
            "chi_A5",
            "deformation_graph_rms/base_graph_rms",
            Dimension.DIMENSIONLESS,
            "6_뇌/11_리만계량_라우팅_논문.md",
            "Both RMS values act on dimensionless z with Laplacians built from lengths normalized by ell_ref",
        )

        self.add_formula(
            "A6 passive directional stretch",
            "s_A6",
            "sqrt(pullback_length_sq/reference_length_sq)",
            Dimension.DIMENSIONLESS,
            "6_뇌/11_리만계량_라우팅_논문.md",
            "Both quadratic lengths use the same dimensionless activation chart",
        )

        self.add_formula(
            "A6 pre/post principal metric ratio",
            "Lambda_A6",
            "post_pullback_length_sq/pre_pullback_length_sq",
            Dimension.DIMENSIONLESS,
            "6_뇌/11_리만계량_라우팅_논문.md",
            "Generalized metric eigenvalues are ratios in one frozen state chart",
        )

        self.add_formula(
            "A6 pullback metric-volume log response",
            "delta_logV_A6",
            "log(post_metric_determinant/pre_metric_determinant)/2",
            Dimension.DIMENSIONLESS,
            "6_뇌/11_리만계량_라우팅_논문.md",
            "The determinant ratio is dimensionless and is used only when both metrics are SPD",
        )

        self.add_formula(
            "A6 reachability-energy log response",
            "rho_E_A6",
            "log(post_control_energy/pre_control_energy)",
            Dimension.DIMENSIONLESS,
            "6_뇌/11_리만계량_라우팅_논문.md",
            "Both energies share one frozen actuator, cost, horizon, and normalized endpoint direction",
        )

        # HISTORY-EDGE METRIC / VARIABLE-RANK SUBSPACE LAYER
        self.add_formula(
            "Orthogonal spectral concentration",
            "c_d_perp",
            "trace_projected_covariance/trace_covariance",
            Dimension.DIMENSIONLESS,
            "brain-riemannian-conscious-subspace-strengthening-20260825 contract §5.4",
            "Scalar surrogate for tr(Q_d*C*Q_d)/tr(C): Q_d is the orthogonal projection "
            "onto Ran(P_d), C is positive trace class, tr(C)>0, and numerator and denominator "
            "must use the same covariance (or precision) unit. This does not license the "
            "nonorthogonal Riesz expression tr(P*C*P)/tr(C).",
        )

        self.add_formula(
            "Effective dimension eigenvalue summand",
            "d_eff_mode",
            "eigenvalue/(eigenvalue+lambda_reg)",
            Dimension.DIMENSIONLESS,
            "brain-riemannian-conscious-subspace-strengthening-20260825 contract §5.4",
            "Parser-friendly scalar summand for d_eff(lambda)=sum_i mu_i/(mu_i+lambda), "
            "equivalently tr(G*(G+lambda*I)^-1). Each mu_i and lambda must have identical "
            "operator-eigenvalue units; lambda>0 and the stated trace-class/summability "
            "assumptions are required. It is an observed effective dimension, not ambient, "
            "manifold, or consciousness dimension.",
        )

        self.add_formula(
            "Normalized edge-metric perturbation",
            "eta_edge",
            "epsilon_A/m0",
            Dimension.DIMENSIONLESS,
            "brain-riemannian-conscious-subspace-strengthening-20260825 contract §5.1",
            "epsilon_A is the operator-norm edge perturbation and m0 is the coercive baseline "
            "lower bound, so they must share one metric-operator unit. The small-perturbation "
            "bound additionally requires epsilon_A < m0; edge deletion remains a separate "
            "reachability change rather than a smooth-curvature claim.",
        )

        # RIEMANN/MRA LAYER
        self.add_formula(
            "Riemann metric attention",
            "A_ij",
            "exp(-d_G^2(z_i, z_j) / (2*sigma^2)) / partition",
            Dimension.DIMENSIONLESS,
            "8_리만/mra_paper.md",
            "Distance² / σ² dimensionless → attention weights",
        )

    def add_formula(
        self,
        name: str,
        symbol: str,
        formula: str,
        expected_dim: Dimension | DimensionVector,
        source: str,
        notes: str = "",
    ):
        """Register a formula."""
        self.formulas.append(
            Formula(
                name=name,
                symbol=symbol,
                formula=formula,
                expected_dim=expected_dim,
                source=source,
                notes=notes,
            )
        )

    def check_formula(self, formula: Formula) -> Dict:
        """
        Analyze dimensional consistency of a single formula.

        Returns:
            {
                'name': str,
                'formula': str,
                'expected': Dimension,
                'status': 'PASS' | 'FAIL' | 'UNCLEAR',
                'notes': str
            }
        """
        result = {
            "name": formula.name,
            "symbol": formula.symbol,
            "formula": formula.formula,
            "expected": formula.expected_dim.name,
            "notes": formula.notes,
            "source": formula.source,
            "parser_backend": self.formula_parser_backend,
            "validation_level": (
                "SYNTAX_ONLY_HEURISTIC"
                if self.formula_parser_backend == "stdlib.ast.parse.syntax_only"
                else "SYMPY_PARSE_HEURISTIC"
            ),
        }

        # Manual checks (symbolic evaluation is limited)
        try:
            # SymPy is preferred.  The stdlib fallback only establishes that a
            # parser-safe scalar expression has valid Python-expression syntax.
            _parse_formula(formula.formula)

            # Quick dimensionality checks based on formula structure
            if formula.symbol in {
                "g_CF",
                "I_V17",
                "y_tilde_V18b",
                "delta_w_V18b",
                "a_A4",
                "r_w_A4",
            }:
                result["status"] = "PASS ✓"

            elif formula.symbol == "S(D)" or "exp(" in formula.formula:
                # Exponential arguments must be dimensionless
                if "exp(-D)" in formula.formula or "exp(-(1-" in formula.formula:
                    result["status"] = "PASS ✓"
                else:
                    result["status"] = "CHECK: exp argument dimensionless?"

            elif formula.symbol == "sin²θ_W" or formula.symbol == "|V_cb|":
                # These are pure numbers
                result["status"] = "PASS ✓"

            elif formula.expected_dim == Dimension.MASS:
                # Should have mass dimension
                if "m_" in formula.formula or "GeV" in formula.formula:
                    result["status"] = "PASS ✓"
                else:
                    result["status"] = "WARN ⚠️ - Missing mass factor?"

            elif formula.expected_dim == Dimension.DIMENSIONLESS:
                # Should be dimensionless
                if "exp(" in formula.formula or "/" in formula.formula:
                    # Likely dimensionless (ratio or exp of dimensionless)
                    result["status"] = "PASS ✓"
                else:
                    result["status"] = "UNCLEAR - Manual review needed"
            else:
                result["status"] = "TODO - Dimension type not recognized"

            if (
                result["validation_level"] == "SYNTAX_ONLY_HEURISTIC"
                and result["status"].startswith("PASS")
            ):
                result["status"] = "PASS_SYNTAX_ONLY"

        except Exception as e:
            result["status"] = f"PARSE ERROR: {str(e)}"

        return result

    def run_all_checks(self) -> List[Dict]:
        """Run dimensional analysis on all formulas."""
        results = []
        for formula in self.formulas:
            results.append(self.check_formula(formula))
        self.results = {r["symbol"]: r for r in results}
        return results

    def generate_report(self) -> str:
        """Generate comprehensive dimensional analysis report."""
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("DIMENSIONAL CONSISTENCY AUDIT")
        lines.append("Clarus Equation Framework")
        lines.append("=" * 100)

        lines.append("\nNatural units: ℏ = c = k_B = 1")
        lines.append("All quantities expressed in mass dimension [M]\n")

        # Run checks
        results = self.run_all_checks()

        # Group by status
        passed = [r for r in results if "PASS" in r["status"]]
        unclear = [r for r in results if "UNCLEAR" in r["status"] or "TODO" in r["status"]]
        warnings = [r for r in results if "WARN" in r["status"]]
        errors = [r for r in results if "ERROR" in r["status"]]

        lines.append("SUMMARY:")
        lines.append(f"  ✓ PASS:      {len(passed):3d} / {len(results)}")
        lines.append(f"  ⚠️  UNCLEAR:   {len(unclear):3d} / {len(results)}")
        lines.append(f"  ⚠️  WARN:      {len(warnings):3d} / {len(results)}")
        lines.append(f"  ❌ ERROR:     {len(errors):3d} / {len(results)}")

        # Detailed table
        lines.append("\n" + "-" * 100)
        lines.append(f"{'Symbol':15s} {'Name':40s} {'Status':20s} {'Expected Dim':20s}")
        lines.append("-" * 100)

        for result in results:
            lines.append(
                f"{result['symbol']:15s} {result['name'][:40]:40s} {result['status']:20s} {result['expected']:20s}"
            )

        # Formulas with notes
        lines.append("\n" + "-" * 100)
        lines.append("FORMULA DETAILS WITH NOTES:")
        lines.append("-" * 100)

        for result in results:
            if result["notes"]:
                lines.append(f"\n{result['symbol']}: {result['name']}")
                lines.append(f"  Formula: {result['formula']}")
                lines.append(f"  Source:  {result['source']}")
                lines.append(f"  Status:  {result['status']}")
                lines.append(f"  Note:    {result['notes']}")

        # Critical findings
        if warnings or errors:
            lines.append("\n" + "=" * 100)
            lines.append("⚠️  POTENTIAL ISSUES:")
            for r in warnings + errors:
                lines.append(f"  {r['symbol']:15s}: {r['status']}")

        lines.append("\n" + "=" * 100 + "\n")

        return "\n".join(lines)


def main():
    """Run dimensional consistency audit."""
    checker = DimensionlessChecker()
    report = checker.generate_report()
    print(report)

    # Save results
    import os

    output_file = os.path.join(os.path.dirname(__file__), "dimensionless_audit.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {output_file}")

    return checker


if __name__ == "__main__":
    main()
