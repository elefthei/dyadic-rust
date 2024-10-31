# dyadic**dyadic** is a Rust library for performing symbolic algebra with dyadic rational numbers. Dyadic rationals, or binary rationals, are numbers that can be expressed as fractions with a power of two as the denominator (e.g., `1/2`, `3/2`, `3/8`). These numbers have finite binary representations, making them ideal for precise approximations in computer science and mathematics.

### Features

- **Arithmetic Operations**: Supports addition, subtraction, and multiplication, which maintain closure within the dyadic rational ring.
- **Division by Powers of Two**: Includes division by powers of two, ensuring the result remains within the set of dyadic rationals.
- **Exact Fractional Representation**: Represents dyadic numbers in their exact fractional form to avoid rounding errors.
- **Simple API**: Provides a clear interface for algebraic operations on dyadic numbers, designed for ease of use in symbolic calculations.

### Mathematical Background

Dyadic rationals form a ring, closed under addition, subtraction, multiplication, and division by powers of two. These properties make dyadic rationals valuable in applications requiring precise, finite representations, including numerical analysis, cryptography, and formal verification.

