import pytest
from unittest.mock import patch, Mock
from utils import map_threaded, extract_xml_tag


class TestMapThreaded:
    """Test cases for the map_threaded function."""

    def test_basic_functionality(self):
        """Test basic mapping functionality with simple function."""

        def square(x):
            return x * x

        numbers = [1, 2, 3, 4, 5]
        result = map_threaded(square, numbers, 2)
        expected = [1, 4, 9, 16, 25]

        assert result == expected

    def test_with_additional_args(self):
        """Test map_threaded with additional positional arguments."""

        def add(x, y):
            return x + y

        numbers = [1, 2, 3, 4]
        result = map_threaded(add, numbers, 2, False, 10)
        expected = [11, 12, 13, 14]

        assert result == expected

    def test_with_kwargs(self):
        """Test map_threaded with keyword arguments."""

        def multiply(x, factor=1):
            return x * factor

        numbers = [1, 2, 3, 4]
        result = map_threaded(multiply, numbers, 2, factor=3)
        expected = [3, 6, 9, 12]

        assert result == expected

    def test_with_args_and_kwargs(self):
        """Test map_threaded with both args and kwargs."""

        def complex_operation(x, offset, multiplier=2):
            return (x + offset) * multiplier

        numbers = [1, 2, 3]
        result = map_threaded(complex_operation, numbers, 2, False, 5, multiplier=3)
        expected = [18, 21, 24]  # (1+5)*3, (2+5)*3, (3+5)*3

        assert result == expected

    def test_empty_iterable(self):
        """Test map_threaded with empty input."""

        def dummy(x):
            return x

        result = map_threaded(dummy, [], 2)
        assert result == []

    def test_single_item(self):
        """Test map_threaded with single item."""

        def square(x):
            return x * x

        result = map_threaded(square, [5], 1)
        assert result == [25]

    def test_order_preservation(self):
        """Test that output order matches input order even with threading."""
        import time

        def slow_process(x):
            # Simulate varying processing time
            time.sleep(0.01 * (5 - x))  # Later items process faster
            return x * 10

        numbers = [1, 2, 3, 4, 5]
        result = map_threaded(slow_process, numbers, 3)
        expected = [10, 20, 30, 40, 50]

        assert result == expected

    def test_exception_handling(self):
        """Test that exceptions in worker functions are properly raised."""

        def failing_function(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2

        numbers = [1, 2, 3, 4]

        with pytest.raises(ValueError, match="Test error"):
            map_threaded(failing_function, numbers, 2)

    @patch("utils.tqdm")
    def test_progress_bar_true(self, mock_tqdm):
        """Test progress bar functionality when show_progress=True."""
        mock_pbar = Mock()
        mock_tqdm.return_value = mock_pbar

        def simple_func(x):
            return x

        numbers = [1, 2, 3]
        map_threaded(simple_func, numbers, 2, show_progress=True)

        mock_tqdm.assert_called_once_with(total=3, desc="Processing")
        assert mock_pbar.update.call_count == 3
        mock_pbar.close.assert_called_once()

    @patch("utils.tqdm")
    def test_progress_bar_custom_desc(self, mock_tqdm):
        """Test progress bar with custom description."""
        mock_pbar = Mock()
        mock_tqdm.return_value = mock_pbar

        def simple_func(x):
            return x

        numbers = [1, 2]
        map_threaded(simple_func, numbers, 2, show_progress="Custom task")

        mock_tqdm.assert_called_once_with(total=2, desc="Custom task")
        mock_pbar.close.assert_called_once()

    def test_max_concurrency_limits(self):
        """Test that max_concurrency parameter works correctly."""
        import threading

        max_threads = []

        def track_threads(x):
            max_threads.append(threading.active_count())
            return x

        numbers = list(range(10))
        map_threaded(track_threads, numbers, 2)

        assert len(max_threads) == 10


class TestExtractXmlTag:
    """Test cases for the extract_xml_tag function."""

    def test_simple_tag_extraction(self):
        """Test extracting a simple XML tag."""
        content = "<root><name>John Doe</name></root>"
        result = extract_xml_tag(content, "name")
        assert result == "John Doe"

    def test_nested_tag_extraction(self):
        """Test extracting from nested XML structure."""
        content = "<root><person><name>Jane Smith</name><age>30</age></person></root>"
        result = extract_xml_tag(content, "name")
        assert result == "Jane Smith"

    def test_tag_with_attributes(self):
        """Test extracting tag with attributes."""
        content = '<root><message type="info">Hello World</message></root>'
        result = extract_xml_tag(content, "message")
        assert result == "Hello World"

    def test_tag_not_found(self):
        """Test behavior when tag is not found."""
        content = "<root><name>John</name></root>"
        result = extract_xml_tag(content, "missing")
        assert result is None

    def test_empty_tag(self):
        """Test extracting empty tag."""
        content = "<root><empty></empty></root>"
        result = extract_xml_tag(content, "empty")
        assert result == ""

    def test_self_closing_tag(self):
        """Test extracting self-closing tag."""
        content = "<root><image src='test.jpg'/></root>"
        result = extract_xml_tag(content, "image")
        assert result == ""

    def test_tag_with_whitespace(self):
        """Test extracting tag with whitespace content."""
        content = "<root><text>  Hello World  </text></root>"
        result = extract_xml_tag(content, "text")
        assert result == "  Hello World  "

    def test_html_content(self):
        """Test extracting from HTML-like content."""
        content = "<html><body><p>This is a paragraph.</p></body></html>"
        result = extract_xml_tag(content, "p")
        assert result == "This is a paragraph."

    def test_multiple_same_tags(self):
        """Test that only first occurrence is returned."""
        content = "<root><item>First</item><item>Second</item></root>"
        result = extract_xml_tag(content, "item")
        assert result == "First"

    def test_malformed_xml(self):
        """Test with malformed XML."""
        content = "<root><unclosed>Some text</root>"
        result = extract_xml_tag(content, "unclosed")
        assert result == "Some text"

    def test_empty_content(self):
        """Test with empty content."""
        result = extract_xml_tag("", "tag")
        assert result is None

    def test_tag_with_cdata(self):
        """Test extracting tag with CDATA section."""
        content = "<root><script><![CDATA[alert('hello');]]></script></root>"
        result = extract_xml_tag(content, "script")
        assert result == "<![CDATA[alert('hello');]]>"

    def test_tag_with_nested_html(self):
        """Test extracting tag containing nested HTML."""
        content = "<root><content><b>Bold</b> and <i>italic</i> text</content></root>"
        result = extract_xml_tag(content, "content")
        assert result == "Bold and italic text"
