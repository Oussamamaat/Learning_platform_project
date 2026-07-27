import json
from app.main import app

def export_spec():
    # FastAPI builds the schema on request, we trigger it manually
    openapi_schema = app.openapi()
    
    # Save as openapi.json
    output_path = "openapi.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
    print(f"Successfully exported OpenAPI schema to {output_path}")

if __name__ == "__main__":
    export_spec()
