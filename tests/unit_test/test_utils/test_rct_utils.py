"""
Random Control Trial API

RCT Utils.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import os
import sys
import pytest
import pandas as pd
from datetime import date

sys.path.append('src/')


from src.utils.rct_utils import RCTTools


class TestRCTTools:
    @staticmethod
    def create_design_table(num_columns):
        return pd.DataFrame(columns=[f"column_{i}" for i in range(num_columns)])

    @pytest.mark.skip
    @pytest.mark.rct_api
    def test_check_input_layout_less_than_12_columns(self):
        design_table = self.create_design_table(10)
        output_table = RCTTools.check_input_layout(design_table)
        assert output_table.shape[1] == design_table.shape[1] + 10

    @pytest.mark.skip
    @pytest.mark.rct_api
    def test_check_input_layout_equal_to_12_columns(self):
        design_table = self.create_design_table(12)
        output_table = RCTTools.check_input_layout(design_table)
        assert output_table.shape[1] == 12

    @pytest.mark.skip
    @pytest.mark.rct_api
    def test_check_input_layout_greater_than_12_columns(self):
        design_table = self.create_design_table(14)
        output_table = RCTTools.check_input_layout(design_table)
        assert output_table.shape[1] == 14

    @pytest.mark.skip
    @pytest.mark.rct_api
    def test_check_input_layout_empty_table(self):
        design_table = pd.DataFrame()
        output_table = RCTTools.check_input_layout(design_table)
        assert output_table.shape[1] == 10

    @pytest.mark.skip
    @pytest.mark.rct_api
    def test_check_input_layout_no_changes(self):
        design_table = self.create_design_table(15)
        output_table = RCTTools.check_input_layout(design_table)
        assert output_table.equals(design_table)

    @pytest.mark.skip
    @pytest.mark.rct_api
    def test_format_dates(self):
        experiment = {
            'start_date': date(2023, 7, 1),
            'end_date': date(2023, 7, 10),
            'created_date': date(2023, 6, 15),
            'invalid_date': '2023-07-01',
        }
        dates_list = ['start_date', 'end_date', 'created_date', 'invalid_date']
        date_format = "%Y-%m-%d"

        result = RCTTools.format_dates(experiment, dates_list, date_format)

        assert result == {
            'start_date': '2023-07-01',
            'end_date': '2023-07-10',
            'created_date': '2023-06-15',
            'invalid_date': '2023-07-01',  # Should remain unchanged
        }


