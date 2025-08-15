"""Not currently used but can be useful for testing."""

from naturalproofs_models import TheoremSample  # type: ignore

_TEST_THEOREM_SAMPLES: list[TheoremSample] = [
    TheoremSample(
        theorem="2 + 2 = 4",
        proof="Adding 2 to 2 yields 4, as per elementary arithmetic. Therefore, 2 + 2 = 4.",
        subtly_incorrect_proof="Adding 2 to 2 sometimes yields 5 due to rounding errors in fractions, so 2 + 2 = 5.",
        difficulty_justification="Trivial arithmetic fact.",
        difficulty_score=1,
        brainstorming="",
    ),
    TheoremSample(
        theorem="The interior angles of a triangle sum to 180 degrees",
        proof=(
            "Draw a line parallel to one side of the triangle through the opposite vertex. "
            "The alternate interior angles show that the three angles add up to a straight line, 180°."
        ),
        subtly_incorrect_proof=(
            "Since a square has interior angles summing to 360°, a triangle (with one less side) must have 240° of interior angle. "
            "Therefore, 180° is incorrect."
        ),
        difficulty_justification="Well-known geometric fact.",
        difficulty_score=1,
        brainstorming="",
    ),
    TheoremSample(
        theorem="1 is not a prime number",
        proof="By definition, a prime number has exactly two positive divisors, 1 and itself. The number 1 has only one positive divisor, so it is not prime.",
        subtly_incorrect_proof="1 is prime because it divides itself exactly once, satisfying the definition of prime numbers.",
        difficulty_justification="Definition-based fact.",
        difficulty_score=1,
        brainstorming="",
    ),
    TheoremSample(
        theorem="√2 is irrational",
        proof=(
            "Assume √2 = a/b in lowest terms. Then 2 = a²/b², so a² = 2b², hence a is even. "
            "Let a = 2k, so 4k² = 2b² and b is even, contradicting lowest terms. Therefore √2 is irrational."
        ),
        subtly_incorrect_proof="√2 = 1.414… which is a terminating decimal when written to enough places, hence rational.",
        difficulty_justification="Classic proof by contradiction taught in school.",
        difficulty_score=1,
        brainstorming="",
    ),
    TheoremSample(
        theorem="There are infinitely many prime numbers",
        proof=(
            "Assume finitely many primes p₁,…,pₙ. Let N = p₁⋯pₙ + 1. "
            "N is not divisible by any pᵢ, so it is prime or divisible by a prime not in the list, contradicting finiteness."
        ),
        subtly_incorrect_proof="The largest prime known is currently 2^82,589,933−1, so primes are finite.",
        difficulty_justification="Euclid’s classic argument.",
        difficulty_score=1,
        brainstorming="",
    ),
]
