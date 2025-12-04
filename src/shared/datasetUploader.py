"""
Módulo para subir datasets procesados a un servidor externo.
Incluye funciones para subir archivos individuales y registrar datasets.
"""

import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os


class DatasetUploader:
    """
    Clase para subir datasets procesados a un servidor externo.
    """

    def __init__(self, base_url: str):
        """
        Inicializa el uploader con la URL base del servidor.

        Args:
            base_url: URL base del servidor (ej: https://example.com:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.upload_folder_endpoint = f"{self.base_url}/upload-folder"
        self.upload_dataset_endpoint = f"{self.base_url}/upload-dataset"

    def upload_file(self, file_path: str, relative_path: str) -> Tuple[bool, str]:
        """
        Sube un archivo individual al endpoint /upload-folder.

        Args:
            file_path: Ruta absoluta del archivo a subir
            relative_path: Ruta relativa del archivo (como se guardará en el servidor)

        Returns:
            Tupla (éxito: bool, mensaje: str)
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return False, f"Archivo no encontrado: {file_path}"

            # ===== FIX: Preparar FormData exactamente como el JS que funciona =====
            # JavaScript hace:
            #   formData.append("files", file);
            #   formData.append("paths", file.webkitRelativePath);
            with open(file_path, 'rb') as f:
                # IMPORTANTE: Usar relative_path como nombre del archivo para preservar estructura
                # Esto incluye dataset_name/sub-01/file.npy, no solo file.npy
                files = {'files': (relative_path, f)}
                # paths debe ser la ruta relativa completa (equivalente a webkitRelativePath)
                data = {'paths': relative_path}

                # Hacer POST request (sin content-type explícito, requests lo maneja)
                response = requests.post(
                    self.upload_folder_endpoint,
                    files=files,
                    data=data,
                    timeout=60  # Aumentar timeout a 60s
                )

                if response.status_code == 200:
                    return True, f"✅ Subido: {relative_path}"
                else:
                    error_text = response.text if response.text else "Sin mensaje de error"
                    return False, f"❌ Error {response.status_code} ({error_text[:100]}): {relative_path}"

        except requests.exceptions.Timeout:
            return False, f"⏱️ Timeout (60s): {relative_path}"
        except requests.exceptions.RequestException as e:
            return False, f"❌ Error de conexión: {str(e)}"
        except Exception as e:
            return False, f"❌ Error inesperado: {str(e)}"

    def upload_all_files_in_folder(self, folder_path: str, base_folder: str, percentage: int = 100) -> Dict:
        """
        Sube todos los archivos de una carpeta recursivamente.

        Args:
            folder_path: Ruta de la carpeta a subir (ej: Aux/dataset_name/)
            base_folder: Carpeta base para calcular rutas relativas (ej: Aux/)
            percentage: Porcentaje de archivos a subir (0-100, default: 100)
                       0 = no subir nada (útil para deshabilitar upload)

        Returns:
            Dict con estadísticas de subida:
            {
                "total": int,
                "uploaded": int,
                "failed": int,
                "errors": List[str]
            }
        """
        import random
        import os

        # ===== CASO ESPECIAL: percentage == 0 → NO subir nada =====
        if percentage == 0:
            print(f"\n[UPLOAD] ⚠️ Porcentaje 0%: No se subirán archivos")
            return {
                "total": 0,
                "uploaded": 0,
                "failed": 0,
                "errors": []
            }

        # ===== FIX: Asegurar que base_folder se resuelva desde el mismo directorio que folder_path =====
        # Convertir folder_path a Path absoluto
        folder_path_obj = Path(folder_path)
        if not folder_path_obj.is_absolute():
            folder_path_obj = folder_path_obj.resolve()

        # Para base_folder, si es relativo, resolverlo desde el mismo contexto
        base_folder_obj = Path(base_folder)
        if not base_folder_obj.is_absolute():
            # Si base_folder es relativo (ej: "Aux"), resolverlo desde el cwd
            base_folder_obj = base_folder_obj.resolve()

        folder_path = folder_path_obj
        base_folder = base_folder_obj

        print(f"\n[UPLOAD] 🔍 DEBUG - Rutas resueltas:")
        print(f"[UPLOAD] 🔍 folder_path absoluto: {folder_path}")
        print(f"[UPLOAD] 🔍 base_folder absoluto: {base_folder}")

        stats = {
            "total": 0,
            "uploaded": 0,
            "failed": 0,
            "errors": []
        }

        if not folder_path.exists():
            stats["errors"].append(f"Carpeta no encontrada: {folder_path}")
            return stats

        # Obtener todos los archivos recursivamente
        all_files = list(folder_path.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]

        # Filtrar archivos dentro de carpetas "Events" (archivos intermedios/cache)
        all_files = [f for f in all_files if "/Events/" not in str(f) and "\\Events\\" not in str(f)]

        total_files = len(all_files)
        stats["total"] = total_files

        # ===== SELECCIONAR PORCENTAJE DE ARCHIVOS =====
        if percentage < 100:
            num_files_to_upload = max(1, int(total_files * percentage / 100))
            # Selección aleatoria con seed para reproducibilidad
            random.seed(42)
            all_files = random.sample(all_files, num_files_to_upload)

            print(f"\n[UPLOAD] 📊 Total de archivos en dataset: {total_files}")
            print(f"[UPLOAD] 📊 Porcentaje seleccionado: {percentage}%")
            print(f"[UPLOAD] 📤 Archivos a subir: {len(all_files)} de {total_files}")
        else:
            print(f"\n[UPLOAD] 📤 Subiendo TODOS los archivos: {total_files}")

        print(f"[UPLOAD] 📁 Carpeta base: {base_folder}")
        print(f"[UPLOAD] 📁 Carpeta dataset: {folder_path}")

        for file_path in all_files:
            # ===== FIX: Calcular ruta relativa INCLUYENDO el nombre del dataset =====
            # La ruta debe ser: dataset_name/sub-01/file.npy (NO solo sub-01/file.npy)
            try:
                relative_path = str(file_path.relative_to(base_folder))

                # Debug: mostrar primera ruta para verificar
                if stats["uploaded"] == 0 and stats["failed"] == 0:
                    print(f"[UPLOAD] 🔍 Ejemplo de ruta relativa: {relative_path}")
                    print(f"[UPLOAD] 🔍 Archivo absoluto: {file_path}")
                    print(f"[UPLOAD] 🔍 Base folder: {base_folder}")

            except ValueError as e:
                # Si no es relativo a base_folder, calcular desde la carpeta dataset
                print(f"[UPLOAD] ⚠️ ValueError al calcular ruta relativa para: {file_path}")
                print(f"[UPLOAD] ⚠️ Base folder era: {base_folder}")
                print(f"[UPLOAD] ⚠️ Error: {e}")
                # Usar el nombre completo desde folder_path.parent
                relative_path = str(file_path.relative_to(folder_path.parent))
                print(f"[UPLOAD] 🔧 Ruta relativa calculada: {relative_path}")

            success, message = self.upload_file(str(file_path), relative_path)

            if success:
                stats["uploaded"] += 1
                print(f"  {message}")
            else:
                stats["failed"] += 1
                stats["errors"].append(message)
                print(f"  {message}")

        print(f"\n[UPLOAD] ✅ Subida completada: {stats['uploaded']}/{stats['total']} archivos")
        if stats["failed"] > 0:
            print(f"[UPLOAD] ⚠️ Fallos: {stats['failed']}")

        return stats

    def register_dataset(self, foldername: str, path: str, desc: str = "", max_retries: int = 10) -> Tuple[bool, str]:
        """
        Registra el dataset en el servidor mediante /upload-dataset.
        Incluye retry logic con hasta max_retries intentos.

        Args:
            foldername: Nombre de la carpeta del dataset
            path: Ruta del dataset
            desc: Descripción opcional del dataset
            max_retries: Número máximo de reintentos (default: 10)

        Returns:
            Tupla (éxito: bool, mensaje: str)
        """
        import time

        for attempt in range(1, max_retries + 1):
            try:
                payload = {
                    "foldername": foldername,
                    "path": path,
                    "desc": desc
                }

                response = requests.post(
                    self.upload_dataset_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code == 200:
                    if attempt > 1:
                        print(f"[REGISTER] ✅ Registro exitoso en intento {attempt}/{max_retries}")
                    return True, f"✅ Dataset '{foldername}' registrado exitosamente"
                else:
                    error_msg = f"❌ Error {response.status_code} al registrar dataset"

                    # Si no es el último intento, reintentar
                    if attempt < max_retries:
                        wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                        print(f"[REGISTER] ⚠️ Intento {attempt}/{max_retries} falló (status {response.status_code})")
                        print(f"[REGISTER] 🔄 Reintentando en {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                    return False, error_msg

            except requests.exceptions.Timeout:
                error_msg = "⏱️ Timeout al registrar dataset"

                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 30)
                    print(f"[REGISTER] ⚠️ Intento {attempt}/{max_retries} falló (timeout)")
                    print(f"[REGISTER] 🔄 Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                return False, error_msg

            except requests.exceptions.RequestException as e:
                error_msg = f"❌ Error de conexión: {str(e)}"

                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 30)
                    print(f"[REGISTER] ⚠️ Intento {attempt}/{max_retries} falló (conexión)")
                    print(f"[REGISTER] 🔄 Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                return False, error_msg

            except Exception as e:
                error_msg = f"❌ Error inesperado: {str(e)}"

                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 30)
                    print(f"[REGISTER] ⚠️ Intento {attempt}/{max_retries} falló (error inesperado)")
                    print(f"[REGISTER] 🔄 Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                return False, error_msg

        # Esto solo se alcanza si todos los reintentos fallaron
        return False, f"❌ Registro falló después de {max_retries} intentos"

    def upload_dataset_complete(
        self,
        dataset_name: str,
        aux_folder: str = "Aux",
        description: str = "",
        percentage: int = 100
    ) -> Dict:
        """
        Sube un dataset completo: archivos + registro.

        Este método:
        1. Sube todos los archivos de Aux/{dataset_name}/ al servidor
        2. Registra el dataset en el servidor

        Args:
            dataset_name: Nombre del dataset
            aux_folder: Carpeta raíz de Aux (default: "Aux")
            description: Descripción del dataset
            percentage: Porcentaje de archivos a subir (1-100, default: 100)

        Returns:
            Dict con resultado completo:
            {
                "success": bool,
                "upload_stats": Dict,
                "registration": Tuple[bool, str],
                "message": str
            }
        """
        result = {
            "success": False,
            "upload_stats": None,
            "registration": None,
            "message": ""
        }

        # 1. Subir archivos
        dataset_path = Path(aux_folder) / dataset_name

        if not dataset_path.exists():
            result["message"] = f"❌ No existe la carpeta: {dataset_path}"
            return result

        print(f"\n{'='*60}")
        print(f"🚀 SUBIENDO DATASET: {dataset_name}")
        print(f"{'='*60}")

        upload_stats = self.upload_all_files_in_folder(
            str(dataset_path),
            aux_folder,
            percentage  # ← Pasar porcentaje
        )
        result["upload_stats"] = upload_stats

        # 2. Registrar dataset
        if upload_stats["uploaded"] > 0:
            success, message = self.register_dataset(
                foldername=dataset_name,
                path=dataset_name,
                desc=description or f"Dataset procesado: {dataset_name}"
            )
            result["registration"] = (success, message)
            print(f"\n[REGISTER] {message}")

            if success:
                result["success"] = True
                result["message"] = f"✅ Dataset '{dataset_name}' subido y registrado ({upload_stats['uploaded']} archivos)"
            else:
                result["success"] = False
                result["message"] = f"⚠️ Archivos subidos pero fallo al registrar: {message}"
        else:
            result["success"] = False
            result["message"] = "❌ No se subieron archivos, no se registró el dataset"

        print(f"\n{'='*60}")
        print(f"RESULTADO: {result['message']}")
        print(f"{'='*60}\n")

        return result


def upload_dataset_to_server(
    dataset_name: str,
    server_url: str,
    aux_folder: str = "Aux",
    description: str = "",
    percentage: int = 100
) -> Dict:
    """
    Función helper para subir un dataset completo al servidor.

    Args:
        dataset_name: Nombre del dataset
        server_url: URL del servidor (ej: https://example.com:8000)
        aux_folder: Carpeta raíz de Aux (default: "Aux")
        description: Descripción del dataset
        percentage: Porcentaje de archivos a subir (1-100, default: 100)

    Returns:
        Dict con resultado de la subida
    """
    uploader = DatasetUploader(server_url)
    return uploader.upload_dataset_complete(dataset_name, aux_folder, description, percentage)
