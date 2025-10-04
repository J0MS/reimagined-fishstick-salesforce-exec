"""Unit test for RCT API Middleware."""

import sys
import json
import pytest
sys.path.append('src/')
from fastapi import status
from src.config.middlewares import MiddlewareTools
from src.config.config import APIMetadata

"""Testing TestMiddleware class."""
class TestMiddlewareTools:
    @pytest.mark.rct_api
    def test_dispatch_header_exception_absent_abi_user(self):
        item = "absent_abi_user"
        response = MiddlewareTools.dispatch_header_exception(item)
        expected_content = {'message': 'abi_user missing in headers'}
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response=json.loads(response.body.decode('utf-8'))
        assert response.get("message") == expected_content.get("message")

    @pytest.mark.rct_api
    def test_dispatch_header_exception_absent_abi_team(self):
        item = "absent_abi_team"
        response = MiddlewareTools.dispatch_header_exception(item)
        expected_content = {'message': 'abi_team missing in headers'}
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response=json.loads(response.body.decode('utf-8'))
        assert response.get("message") == expected_content.get("message")

    @pytest.mark.rct_api
    def test_dispatch_header_exception_invalid_abi_team(self):
        item = "invalid_abi_team"
        response = MiddlewareTools.dispatch_header_exception(item)
        expected_content = {'message': 'abi_team not registered'}
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response=json.loads(response.body.decode('utf-8'))
        assert response.get("message") == expected_content.get("message")

    @pytest.mark.rct_api
    def test_dispatch_header_exception_invalid_item(self):
        item = "invalid_item"
        response = MiddlewareTools.dispatch_header_exception(item)
        expected_content = {'message': 'No handler available'}
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response=json.loads(response.body.decode('utf-8'))
        assert response.get("message") == expected_content.get("message")

