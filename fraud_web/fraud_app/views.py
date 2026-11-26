import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
from .ml_service import FraudPredictionService


class IndexView(View):
    def get(self, request):
        service = FraudPredictionService.get_instance()
        context = {
            "metrics_json": json.dumps(service.metrics),
            "samples_json": json.dumps(service.samples),
            "dataset_summary_json": json.dumps(service.dataset_summary),
        }
        return render(request, "fraud_app/index.html", context)


class PredictView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            model_name = data.get("model", "XGBoost")
            features = data.get("features", {})
            if not isinstance(features, dict):
                return JsonResponse({"status": "error", "message": "Trường features phải là object JSON."}, status=400)

            service = FraudPredictionService.get_instance()
            result = service.predict(model_name, features)
            if "error" in result:
                return JsonResponse({"status": "error", "message": result["error"]}, status=400)

            return JsonResponse({"status": "success", "result": result})
        except json.JSONDecodeError:
            return JsonResponse({"status": "error", "message": "Payload JSON không hợp lệ."}, status=400)
        except Exception as exc:
            return JsonResponse({"status": "error", "message": str(exc)}, status=400)
