
class Prediction_Orchestrator:
    def __init__(self, clients: list):
        self.clients = clients
        self.server_model_weights = None

    def fl_predict(self):

        # very last fl round to just predict
        for client in self.clients:
            # empty client.model.current_model_weights to None such that nnUNet's run_training loads model weights from checkpoint (via def maybe_load_checkpoint())
            client.update_model(server_model_weights=None)
            client.fed_round(
                very_last_fl_predict_round=True,
                only_run_validation=True,
            )

        return self.server_model_weights
