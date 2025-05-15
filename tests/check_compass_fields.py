from cohere_compass.models.documents import CompassDocument

print("CompassDocument model fields:")
print(CompassDocument.model_fields)

print("\nCompassDocument schema:")
print(CompassDocument.model_json_schema()) 