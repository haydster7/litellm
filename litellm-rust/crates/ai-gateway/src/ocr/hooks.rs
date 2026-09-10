use litellm_core::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};
use litellm_core::error::Error;
use serde_json::json;

use crate::integrations::custom_guardrail::{
    CustomGuardrailRunner, GuardrailContext, GuardrailError, GuardrailRequest,
};
use crate::integrations::custom_logger::{
    CallType, CallbackTiming, CallbackValue, CustomLoggerRunner, LoggingError, ModelCallDetails,
};
use crate::integrations::types::{
    RequestMetadata, StandardLoggingMetadata, StandardLoggingPayload,
};

pub(crate) struct OcrLifecycleHooks {
    logger_runner: CustomLoggerRunner,
    guardrail_runner: CustomGuardrailRunner,
    request_metadata: RequestMetadata,
}

impl OcrLifecycleHooks {
    pub(crate) fn new(
        logger_runner: CustomLoggerRunner,
        guardrail_runner: CustomGuardrailRunner,
        request_metadata: RequestMetadata,
    ) -> Self {
        Self {
            logger_runner,
            guardrail_runner,
            request_metadata,
        }
    }

    fn standard_logging_payload(
        &self,
        context: &CallLifecycleContext,
        timing: &CallLifecycleTiming,
    ) -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: context.litellm_call_id.clone(),
            litellm_call_id: context.litellm_call_id.clone(),
            call_type: context.call_type.clone(),
            model: context.model.clone(),
            custom_llm_provider: context.custom_llm_provider.clone(),
            response_cost: 0.0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            start_time: timing.start_time,
            end_time: timing.end_time,
            stream: false,
            metadata: StandardLoggingMetadata {
                user_api_key_hash: self.request_metadata.user_api_key_hash.clone(),
                user_api_key_user_id: self.request_metadata.user_api_key_user_id.clone(),
                user_api_key_team_id: self.request_metadata.user_api_key_team_id.clone(),
                ..Default::default()
            },
            messages: None,
        }
    }
}

fn guardrail_context(metadata: &RequestMetadata) -> GuardrailContext {
    GuardrailContext {
        call_type: CallType::Ocr,
        selected_guardrails: Vec::new(),
        metadata: std::collections::HashMap::new(),
        user_api_key_hash: metadata.user_api_key_hash.clone(),
        user_api_key_user_id: metadata.user_api_key_user_id.clone(),
        user_api_key_team_id: metadata.user_api_key_team_id.clone(),
        trace_parent: None,
    }
}

fn guardrail_error_to_core_error(error: GuardrailError) -> Error {
    Error::InvalidRequest(format!("{}: {}", error.kind, error.message))
}

fn core_error_kind(error: &Error) -> &'static str {
    match error {
        Error::Auth(_) => "AuthError",
        Error::InvalidProvider(_) => "InvalidProvider",
        Error::InvalidRequest(_) => "InvalidRequest",
        Error::InvalidType { .. } => "InvalidType",
        Error::MissingField(_) => "MissingField",
        Error::Http { .. } => "HttpError",
        Error::InvalidResponse(_) => "InvalidResponse",
        Error::Network(_) => "NetworkError",
        Error::Connect(_) => "ConnectError",
        Error::Routing(_) => "RoutingError",
        Error::Unsupported(_) => "UnsupportedRequest",
    }
}

impl litellm_core::ocr::hooks::OcrHooks for OcrLifecycleHooks {
    fn has_guardrails(&self) -> bool {
        !self.guardrail_runner.is_empty()
    }

    fn pre_call(
        &self,
        request: litellm_core::ocr::hooks::OcrPreCallRequest,
    ) -> litellm_core::ocr::hooks::OcrHookFuture<'_, litellm_core::ocr::hooks::OcrPreCallRequest>
    {
        Box::pin(async move {
            let context = guardrail_context(&self.request_metadata);
            let payload = GuardrailRequest::new(json!(&request));
            let (changed, _) = self
                .guardrail_runner
                .run_pre_call(&context, payload)
                .await
                .map_err(guardrail_error_to_core_error)?;
            Ok(litellm_core::ocr::wire::decode_pre_call_result(
                request,
                changed.data,
            )?)
        })
    }

    fn during_call(
        &self,
        request: litellm_core::ocr::hooks::OcrDuringCallRequest,
    ) -> litellm_core::ocr::hooks::OcrHookFuture<'_, litellm_core::ocr::hooks::OcrDuringCallRequest>
    {
        Box::pin(async move {
            let context = guardrail_context(&self.request_metadata);
            let payload = GuardrailRequest::new(json!(&request));
            let (changed, _) = self
                .guardrail_runner
                .run_during_call(&context, payload)
                .await
                .map_err(guardrail_error_to_core_error)?;
            Ok(litellm_core::ocr::wire::decode_during_call_result(
                request,
                changed.data,
            )?)
        })
    }

    fn success<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        response: &'a litellm_core::ocr::LiteLLMOcrResponse,
        timing: &'a CallLifecycleTiming,
    ) -> litellm_core::ocr::hooks::OcrLogFuture<'a> {
        Box::pin(async move {
            if self.logger_runner.is_empty() {
                return;
            }
            self.logger_runner
                .async_log_success_event(
                    &ModelCallDetails::from_standard_logging_payload(
                        self.standard_logging_payload(context, timing),
                    ),
                    &CallbackValue::new("ocr", json!(response)),
                    CallbackTiming::new(timing.start_time, timing.end_time),
                )
                .await;
        })
    }

    fn failure<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        error: &'a Error,
        timing: &'a CallLifecycleTiming,
    ) -> litellm_core::ocr::hooks::OcrLogFuture<'a> {
        Box::pin(async move {
            if self.logger_runner.is_empty() {
                return;
            }
            let logging_error = LoggingError {
                message: error.to_string(),
                kind: core_error_kind(error).to_string(),
            };
            let response = CallbackValue::new(
                "error",
                json!({"message":logging_error.message,"kind":logging_error.kind}),
            );
            self.logger_runner
                .async_log_failure_event(
                    &ModelCallDetails::from_standard_logging_payload(
                        self.standard_logging_payload(context, timing),
                    )
                    .with_failure_error(logging_error),
                    Some(&response),
                    CallbackTiming::new(timing.start_time, timing.end_time),
                )
                .await;
        })
    }
}
