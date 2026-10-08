class TrailingStop:

    # ==========================
    # LONG TRAIL
    # ==========================

    def long_trail(
        self,
        current_price,
        stop_loss,
        atr
    ):

        new_stop = max(

            stop_loss,

            current_price - atr

        )

        return new_stop


    # ==========================
    # SHORT TRAIL
    # ==========================

    def short_trail(
        self,
        current_price,
        stop_loss,
        atr
    ):

        new_stop = min(

            stop_loss,

            current_price + atr

        )

        return new_stop