#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple test to verify pytest fix."""

import sys
import io

# 修复编码（pytest安全）
try:
    from tests.is_EUR_work.utils.encoding_utils import setup_utf8_encoding
    setup_utf8_encoding()
except ImportError:
    # Fallback for scripts outside tests directory
    if sys.platform == 'win32' and 'pytest' not in sys.modules:
        try:
            if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding.lower() != 'utf-8':
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass
        try:
            if hasattr(sys.stderr, 'buffer') and sys.stderr.encoding.lower() != 'utf-8':
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass

def test_basic_addition():
    """Test basic addition."""
    assert 1 + 1 == 2

def test_string_concatenation():
    """Test string concatenation."""
    assert "hello" + " " + "world" == "hello world"

def test_list_operations():
    """Test list operations."""
    lst = [1, 2, 3]
    lst.append(4)
    assert lst == [1, 2, 3, 4]

def test_stdout_not_closed():
    """Test that stdout is not closed."""
    assert not sys.stdout.closed
    assert not sys.stderr.closed
    print("Testing output...")
    print("中文测试", file=sys.stderr)
