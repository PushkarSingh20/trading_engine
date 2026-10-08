class ATRRiskEngine:

    def __init__(
        self,
        capital=100000,
        risk_per_trade=0.01,
        atr_multiplier=2,
        target_multiplier=2.0
    ):

        self.capital = capital

        self.risk_per_trade = risk_per_trade

        self.atr_multiplier = atr_multiplier

        self.target_multiplier = target_multiplier


    # ==========================
    # POSITION SIZE
    # ==========================

    def calculate_position_size(
        self,
        atr
    ):

        risk_amount = (

            self.capital
            *
            self.risk_per_trade

        )

        stop_distance = (

            atr
            *
            self.atr_multiplier

        )

        if stop_distance <= 0:

            return 0

        size = (

            risk_amount
            /
            stop_distance

        )

        return int(size)


    # ==========================
    # LONG STOP
    # ==========================

    def long_stop(
        self,
        entry,
        atr
    ):

        return (

            entry
            -
            (
                atr
                *
                self.atr_multiplier
            )
        )


    # ==========================
    # LONG TARGET
    # ==========================

    def long_target(
        self,
        entry,
        atr
    ):

        return (

            entry
            +
            (
                atr
                *
                self.atr_multiplier
                *
                self.target_multiplier
            )
        )


    # ==========================
    # SHORT STOP
    # ==========================

    def short_stop(
        self,
        entry,
        atr
    ):

        return (

            entry
            +
            (
                atr
                *
                self.atr_multiplier
            )
        )


    # ==========================
    # SHORT TARGET
    # ==========================

    def short_target(
        self,
        entry,
        atr
    ):

        return (

            entry
            -
            (
                atr
                *
                self.atr_multiplier
                *
                self.target_multiplier
            )
        )