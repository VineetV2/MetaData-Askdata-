import unittest


class LLMBackendTests(unittest.TestCase):
    def test_make_backend_creates_vllm_openai_compatible_backend(self):
        from llm import OpenAICompatibleBackend, make_backend

        backend = make_backend(
            "vllm",
            model_id="openai/gpt-oss-20b",
            api_base="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            timeout_seconds=77,
            cache=False,
        )

        self.assertIsInstance(backend, OpenAICompatibleBackend)
        self.assertEqual(backend.model_id, "openai/gpt-oss-20b")
        self.assertEqual(backend.api_base, "http://127.0.0.1:8000/v1")
        self.assertEqual(backend.api_key, "EMPTY")
        self.assertEqual(backend.timeout_seconds, 77)

    def test_openai_compatible_backend_has_client_shape(self):
        from llm import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(
            "openai/gpt-oss-20b",
            api_base="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            timeout_seconds=88,
        )

        self.assertTrue(hasattr(backend, "_client"))
        self.assertEqual(backend.timeout_seconds, 88)


if __name__ == "__main__":
    unittest.main()
