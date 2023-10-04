from pathlib import Path


class Error(Exception):
    pass


class DirectoryNotFoundError(Error):
    def __init__(self, path_to_directory: Path) -> None:
        self._message = f"The directory {path_to_directory} could not be found"
        super().__init__(self._message)


class FileNotFoundError(Error):
    def __init__(self, path_to_file: Path) -> None:
        self._message = f"The requested file {path_to_file} could not be found!"
        super().__init__(self._message)


class TestConfigurationError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnvalidCalibrationDataError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class CalibrationConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnivariateUniformDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnivariateNormalDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MixedDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class BayesianNNError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class FEMConfigurationError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class FEMGeometryConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class FEMProblemConfigError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)
