"""Step 1 verification: validators bite, schemas generate."""
import sys
from pydantic import ValidationError

try:
    from src.models import Claim, SourceAnalysis
    print("✓ Check 1: models import successfully")
except Exception as e:
    print(f"✗ Check 1 FAILED — import error: {e}")
    sys.exit(1)

print("\nCheck 2: feeding Claim a 300-char statement (should raise ValidationError)...")
try:
    Claim(
        statement="x" * 300,
        claim_type="fact",
        confidence="high",
        source_url="https://example.com",
    )
    print("✗ Check 2 FAILED — 300-char statement was accepted.")
    sys.exit(1)
except ValidationError as e:
    first_err = e.errors()[0]
    print(f"✓ Check 2: rejected as expected — {first_err['msg']}")

print("\nCheck 3: generating SourceAnalysis JSON schema...")
try:
    schema = SourceAnalysis.model_json_schema()
    props = schema.get("properties", {})
    assert "core_claims" in props, "missing 'core_claims'"
    assert "credibility_score" in props, "missing 'credibility_score'"
    print(f"✓ Check 3: schema generated with {len(props)} top-level fields")
    print(f"  fields: {list(props.keys())}")
except Exception as e:
    print(f"✗ Check 3 FAILED: {e}")
    sys.exit(1)

print("\n🎉 All three checks passed. Ready for Step 2.")
