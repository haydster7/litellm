use super::{MistralOcrParams, MistralOcrRequest, MistralOcrResponse};
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{LiteLLMOcrResponse, OcrDocument};

pub(crate) fn transform_ocr_request(
    model: &str,
    document: OcrDocument,
    params: &MistralOcrParams,
) -> Result<MistralOcrRequest, OcrRequestError> {
    Ok(MistralOcrRequest {
        model: model.to_string(),
        document,
        params: params.clone(),
    })
}

pub(crate) fn transform_ocr_response(
    model: &str,
    response: MistralOcrResponse,
) -> Result<LiteLLMOcrResponse, OcrResponseError> {
    Ok(LiteLLMOcrResponse {
        pages: response.pages,
        model: response.model.unwrap_or_else(|| model.to_string()),
        document_annotation: response.document_annotation,
        usage_info: response.usage_info,
        object: "ocr".to_string(),
        extra_fields: response.extra_fields,
        provider_native_response: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_includes_supported_params_and_filters_unknown_fields() {
        let options = json!({
            "pages":[0,2],
            "include_image_base64":true,
            "image_limit":2,
            "image_min_size":100,
            "bbox_annotation_format":{"type":"json_schema"},
            "document_annotation_format":{"type":"json_schema"},
            "document_annotation_prompt":"extract",
            "extract_header":true,
            "extract_footer":false,
            "table_format":"html",
            "confidence_scores_granularity":"word",
            "include_blocks":true,
            "id":"req-123",
            "unknown":{}
        });
        let params: MistralOcrParams = serde_json::from_value(options.clone()).unwrap();
        let document: OcrDocument = serde_json::from_value(
            json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
        )
        .unwrap();
        let result =
            serde_json::to_value(transform_ocr_request("model", document, &params).unwrap())
                .unwrap();
        assert_eq!(result["model"], "model");
        for (name, value) in options.as_object().unwrap() {
            if name == "unknown" {
                assert!(result.get(name).is_none());
            } else {
                assert_eq!(&result[name], value);
            }
        }
    }

    #[test]
    fn response_preserves_provider_fields() {
        let response: MistralOcrResponse = serde_json::from_value(json!({
            "pages":[{"index":0,"markdown":"hello","header":"head","confidence_scores":{"mean":0.99}}],
            "model":"returned-model",
            "usage_info":{"pages_processed":1,"future_counter":5},
            "future_response_field":"kept"
        }))
        .unwrap();
        let result = transform_ocr_response("model", response)
            .unwrap()
            .into_json();
        assert_eq!(result["pages"][0]["header"], "head");
        assert_eq!(result["usage_info"]["future_counter"], 5);
        assert_eq!(result["future_response_field"], "kept");
        assert_eq!(result["model"], "returned-model");
    }

    #[test]
    fn response_rejects_null_pages() {
        assert!(serde_json::from_value::<MistralOcrResponse>(json!({"pages":null})).is_err());
    }
}
