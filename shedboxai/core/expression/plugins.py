"""
Plugin system for the expression engine.

This module provides a flexible plugin architecture that allows
extending the expression engine with custom functions and operators.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .evaluator import ExpressionEngine


class ExpressionPlugin(ABC):
    """Abstract base class for all expression engine plugins."""

    def __init__(self, name: str, version: str = "1.0"):
        """
        Initialize a new plugin.

        Args:
            name: Unique plugin identifier (lowercase, no spaces)
            version: Plugin version string
        """
        self.name = name.lower().strip()
        self.version = version
        self._registered_components: Dict[str, Any] = {}

    @abstractmethod
    def register(self, engine: "ExpressionEngine") -> None:
        """
        Register all plugin components with the engine.

        Should register:
        - Functions (engine.register_function())
        - Operators (engine.register_operator())
        - Any other extensions

        Args:
            engine: The ExpressionEngine instance to register with
        """
        pass

    @abstractmethod
    def unregister(self, engine: "ExpressionEngine") -> None:
        """
        Clean up all plugin components from the engine.

        Should unregister:
        - All previously registered functions/operators
        - Any other resources

        Args:
            engine: The ExpressionEngine instance to unregister from
        """
        pass

    def get_registered_components(self) -> Dict[str, Any]:
        """
        Get all registered components for this plugin.

        Returns:
            Dictionary of {component_type: component_list}
        """
        return self._registered_components.copy()


class PluginManager:
    """Manages plugin lifecycle and registration."""

    def __init__(self, engine: "ExpressionEngine"):
        """
        Initialize with parent engine.

        Args:
            engine: The ExpressionEngine instance this manages plugins for
        """
        self.engine = engine
        self._plugins: Dict[str, ExpressionPlugin] = {}
        self.logger = logging.getLogger(__name__)

    def register_plugin(self, plugin: ExpressionPlugin) -> bool:
        """
        Register a new plugin with the engine.

        Args:
            plugin: The plugin instance to register

        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            if plugin.name in self._plugins:
                self.logger.warning(f"Plugin {plugin.name} already registered, replacing")
                self.unregister_plugin(plugin.name)

            plugin.register(self.engine)
            self._plugins[plugin.name] = plugin
            self.logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin.name}: {str(e)}")
            return False

    def unregister_plugin(self, name: str) -> bool:
        """
        Unregister a plugin by name.

        Args:
            name: The name of the plugin to unregister

        Returns:
            True if unregistration succeeded, False otherwise
        """
        try:
            if name in self._plugins:
                self._plugins[name].unregister(self.engine)
                del self._plugins[name]
                self.logger.info(f"Unregistered plugin: {name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to unregister plugin {name}: {str(e)}")
            return False

    def get_plugin(self, name: str) -> Optional[ExpressionPlugin]:
        """
        Get a registered plugin by name.

        Args:
            name: The plugin name to retrieve

        Returns:
            The plugin instance if found, None otherwise
        """
        return self._plugins.get(name.lower())

    def list_plugins(self) -> List[str]:
        """Get names of all registered plugins."""
        return list(self._plugins.keys())

    def unregister_all(self) -> None:
        """Unregister all plugins."""
        for name in list(self._plugins.keys()):
            self.unregister_plugin(name)
