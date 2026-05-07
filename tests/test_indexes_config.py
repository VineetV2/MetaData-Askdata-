import unittest


class IndexConfigTests(unittest.TestCase):
    def test_configure_embeddings_updates_index_settings(self):
        import indexes
        from config import load_config

        old_backend = indexes.EMBEDDING_BACKEND
        old_model = indexes.EMBEDDING_MODEL
        old_device = indexes.EMBEDDING_DEVICE
        old_batch_size = indexes.EMBEDDING_BATCH_SIZE

        cfg = load_config("config.yaml")
        try:
            indexes.configure_embeddings(cfg)

            self.assertEqual(indexes.EMBEDDING_BACKEND, cfg.embeddings.backend)
            self.assertEqual(indexes.EMBEDDING_MODEL, cfg.embeddings.model)
            self.assertEqual(indexes.EMBEDDING_DEVICE, cfg.embeddings.device)
            self.assertEqual(indexes.EMBEDDING_BATCH_SIZE, cfg.embeddings.batch_size)
        finally:
            indexes.EMBEDDING_BACKEND = old_backend
            indexes.EMBEDDING_MODEL = old_model
            indexes.EMBEDDING_DEVICE = old_device
            indexes.EMBEDDING_BATCH_SIZE = old_batch_size


if __name__ == "__main__":
    unittest.main()
