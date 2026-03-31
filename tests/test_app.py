import pytest
import sys
import numpy as np
import pandas as pd
sys.path.append(".")

from utils.data_processor import (
    parse_price_sale, parse_price_rental,
    parse_area, parse_address,
    clean_sale_data, clean_rental_data,
    load_sale_data, load_rental_data
)
from utils.predictor import predict_price

# ========== TEST DATA PROCESSOR ==========

class TestParsePriceSale:
    def test_ty(self):
        assert parse_price_sale("22,96 tỷ") == pytest.approx(22960, rel=0.01)

    def test_trieu(self):
        assert parse_price_sale("500 triệu") == pytest.approx(500, rel=0.01)

    def test_nan(self):
        assert np.isnan(parse_price_sale(None))

    def test_invalid(self):
        assert np.isnan(parse_price_sale("abc"))

    def test_ty_integer(self):
        assert parse_price_sale("12 tỷ") == pytest.approx(12000, rel=0.01)


class TestParsePriceRental:
    def test_trieu_thang(self):
        result = parse_price_rental("26 triệu/tháng")
        assert result == pytest.approx(26, rel=0.01)

    def test_nan(self):
        assert np.isnan(parse_price_rental(None))

    def test_invalid(self):
        assert np.isnan(parse_price_rental("abc"))


class TestParseArea:
    def test_m2(self):
        assert parse_area("75 m²") == pytest.approx(75, rel=0.01)

    def test_decimal(self):
        assert parse_area("87,5 m²") == pytest.approx(87.5, rel=0.01)

    def test_nan(self):
        assert np.isnan(parse_area(None))


class TestParseAddress:
    def test_full_address(self):
        district, city = parse_address("·\nHai Bà Trưng, Hà Nội")
        assert district == "Hai Bà Trưng"
        assert city == "Hà Nội"

    def test_single_part(self):
        district, city = parse_address("Hà Nội")
        assert district == "Hà Nội"
        assert city == "Không rõ"

    def test_nan(self):
        district, city = parse_address(None)
        assert district == "Không rõ"
        assert city == "Không rõ"


class TestCleanData:
    def setup_method(self):
        """Load data thật để test"""
        self.df_sale = clean_sale_data(load_sale_data())
        self.df_rental = clean_rental_data(load_rental_data())

    def test_sale_no_nulls_in_price(self):
        assert self.df_sale["price_million"].isnull().sum() == 0

    def test_sale_no_nulls_in_area(self):
        assert self.df_sale["area_m2"].isnull().sum() == 0

    def test_sale_price_range(self):
        assert self.df_sale["price_million"].min() >= 100
        assert self.df_sale["price_million"].max() <= 200000

    def test_sale_area_range(self):
        assert self.df_sale["area_m2"].min() >= 10
        assert self.df_sale["area_m2"].max() <= 2000

    def test_sale_no_duplicates(self):
        assert self.df_sale["product_id"].duplicated().sum() == 0

    def test_rental_no_nulls_in_price(self):
        assert self.df_rental["price_million"].isnull().sum() == 0

    def test_rental_price_range(self):
        assert self.df_rental["price_million"].min() >= 1
        assert self.df_rental["price_million"].max() <= 5000

    def test_sale_has_data(self):
        assert len(self.df_sale) > 1000

    def test_rental_has_data(self):
        assert len(self.df_rental) > 1000


# ========== TEST PREDICTOR ==========

class TestPredictor:
    def test_sale_prediction_returns_3_values(self):
        predicted, low, high = predict_price(
            "sale", 60, 2, 2, "Hai Bà Trưng", "Hà Nội"
        )
        assert predicted is not None
        assert low is not None
        assert high is not None

    def test_sale_price_range_logical(self):
        predicted, low, high = predict_price(
            "sale", 60, 2, 2, "Hai Bà Trưng", "Hà Nội"
        )
        assert low < predicted < high

    def test_sale_low_is_85_percent(self):
        predicted, low, high = predict_price(
            "sale", 60, 2, 2, "Cầu Giấy", "Hà Nội"
        )
        assert abs(low - predicted * 0.85) < 1

    def test_sale_high_is_115_percent(self):
        predicted, low, high = predict_price(
            "sale", 60, 2, 2, "Cầu Giấy", "Hà Nội"
        )
        assert abs(high - predicted * 1.15) < 1

    def test_larger_area_higher_price(self):
        _, price_small, _ = predict_price(
            "sale", 50, 2, 2, "Cầu Giấy", "Hà Nội"
        )
        _, price_large, _ = predict_price(
            "sale", 150, 2, 2, "Cầu Giấy", "Hà Nội"
        )
        assert price_large > price_small

    def test_rental_prediction_positive(self):
        predicted, low, high = predict_price(
            "rental", 50, 2, 1, "Cầu Giấy", "Hà Nội"
        )
        assert predicted > 0

    def test_unknown_district_no_crash(self):
        """Model không crash khi gặp quận không có trong data"""
        predicted, low, high = predict_price(
            "sale", 60, 2, 2, "Quận Không Tồn Tại", "Hà Nội"
        )
        assert predicted > 0