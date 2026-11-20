//! Every registry's metrics as Prometheus text.

use crate::http::{Body, Response};

pub(crate) fn metrics() -> Response {
    Response {
        status: 200,
        body: Body::Full {
            content_type: "text/plain; version=0.0.4",
            bytes: sensai_engine::metrics::text().into_bytes(),
        },
    }
}
