"""
Script de visualização e análise de datasets HLS (NDVI/EVI)
Permite explorar dados NetCDF de forma interativa
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('TkAgg')  # Force backend antes de importar pyplot
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DatasetViewer:
    """Visualizador interativo de datasets NetCDF"""
    
    def __init__(self, netcdf_path: str):
        self.netcdf_path = Path(netcdf_path)
        self.ds = None
        self.current_time_idx = 0
        
    def load_dataset(self) -> bool:
        """Carrega o dataset NetCDF"""
        try:
            print(f"📂 Carregando dataset: {self.netcdf_path}")
            self.ds = xr.open_dataset(self.netcdf_path)
            
            print("\n" + "="*60)
            print("📊 INFORMAÇÕES DO DATASET")
            print("="*60)
            print(f"📅 Timestamps: {len(self.ds.time)}")
            print(f"📏 Dimensões espaciais: {self.ds.dims['y']} x {self.ds.dims['x']} pixels")
            print(f"🗓️  Período: {self.ds.time.values[0]} até {self.ds.time.values[-1]}")
            print(f"📈 Variáveis: {list(self.ds.data_vars)}")
            
            # Estatísticas
            print("\n" + "="*60)
            print("📈 ESTATÍSTICAS GLOBAIS")
            print("="*60)
            
            for var in ['ndvi', 'evi']:
                if var in self.ds:
                    data = self.ds[var].values
                    print(f"\n{var.upper()}:")
                    print(f"  Min:    {np.nanmin(data):.4f}")
                    print(f"  Max:    {np.nanmax(data):.4f}")
                    print(f"  Média:  {np.nanmean(data):.4f}")
                    print(f"  Mediana: {np.nanmedian(data):.4f}")
                    print(f"  Std:    {np.nanstd(data):.4f}")
                    
                    # Percentual de pixels válidos
                    valid_pixels = np.sum(~np.isnan(data))
                    total_pixels = data.size
                    valid_pct = (valid_pixels / total_pixels) * 100
                    print(f"  Pixels válidos: {valid_pct:.1f}%")
            
            print("="*60)
            
            # DEBUG: Verificar se há dados
            print(f"\n🔍 DEBUG: Shape NDVI = {self.ds.ndvi.shape}")
            print(f"🔍 DEBUG: Shape EVI = {self.ds.evi.shape}")
            print(f"🔍 DEBUG: Primeiro timestamp = {self.ds.time.values[0]}")
            
            # Verificar se há valores não-NaN
            ndvi_sample = self.ds.ndvi.isel(time=0).values
            evi_sample = self.ds.evi.isel(time=0).values
            print(f"🔍 DEBUG: NDVI tem {np.sum(~np.isnan(ndvi_sample))} pixels válidos")
            print(f"🔍 DEBUG: EVI tem {np.sum(~np.isnan(evi_sample))} pixels válidos")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def plot_single_timestamp(self, time_idx: int = 0, save: bool = False):
        """Plota NDVI e EVI para um único timestamp"""
        
        try:
            if time_idx >= len(self.ds.time):
                time_idx = len(self.ds.time) - 1
            
            timestamp = self.ds.time.values[time_idx]
            date_str = np.datetime_as_string(timestamp, unit='D')
            
            ndvi = self.ds.ndvi.isel(time=time_idx).values
            evi = self.ds.evi.isel(time=time_idx).values
            
            print(f"\n🔍 Plotando timestamp {time_idx}: {date_str}")
            print(f"   NDVI shape: {ndvi.shape}, range: [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")
            print(f"   EVI shape: {evi.shape}, range: [{np.nanmin(evi):.3f}, {np.nanmax(evi):.3f}]")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # NDVI
            im1 = ax1.imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1.0, interpolation='nearest')
            ax1.set_title(f'NDVI - {date_str}', fontsize=14, fontweight='bold')
            ax1.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('NDVI', fontsize=12)
            
            # Adicionar estatísticas no NDVI
            ndvi_valid = ndvi[~np.isnan(ndvi)]
            if len(ndvi_valid) > 0:
                stats_text = f'Min: {np.min(ndvi_valid):.3f}\n'
                stats_text += f'Max: {np.max(ndvi_valid):.3f}\n'
                stats_text += f'Média: {np.mean(ndvi_valid):.3f}'
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # EVI
            im2 = ax2.imshow(evi, cmap='RdYlGn', vmin=-0.2, vmax=2.0, interpolation='nearest')
            ax2.set_title(f'EVI - {date_str}', fontsize=14, fontweight='bold')
            ax2.axis('off')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('EVI', fontsize=12)
            
            # Adicionar estatísticas no EVI
            evi_valid = evi[~np.isnan(evi)]
            if len(evi_valid) > 0:
                stats_text = f'Min: {np.min(evi_valid):.3f}\n'
                stats_text += f'Max: {np.max(evi_valid):.3f}\n'
                stats_text += f'Média: {np.mean(evi_valid):.3f}'
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle(f'Timestamp {time_idx + 1}/{len(self.ds.time)}', 
                         fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            if save:
                output_path = self.netcdf_path.parent / f"frame_{time_idx:03d}_{date_str}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"💾 Salvo: {output_path}")
                plt.close()
            else:
                print("🖼️  Mostrando janela... (feche para continuar)")
                plt.show()
                
        except Exception as e:
            print(f"❌ Erro ao plotar: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_time_series(self, y: int = None, x: int = None):
        """Plota série temporal de NDVI/EVI para um pixel específico"""
        
        try:
            if y is None:
                y = self.ds.dims['y'] // 2
            if x is None:
                x = self.ds.dims['x'] // 2
            
            print(f"\n📊 Plotando série temporal do pixel ({y}, {x})")
            
            ndvi_ts = self.ds.ndvi.isel(y=y, x=x).values
            evi_ts = self.ds.evi.isel(y=y, x=x).values
            times = self.ds.time.values
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            
            # NDVI
            ax1.plot(times, ndvi_ts, 'o-', color='green', linewidth=2, markersize=6, label='NDVI')
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_ylabel('NDVI', fontsize=12, fontweight='bold')
            ax1.set_title(f'Série Temporal - Pixel ({y}, {x})', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # EVI
            ax2.plot(times, evi_ts, 'o-', color='darkgreen', linewidth=2, markersize=6, label='EVI')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('EVI', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Data', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            print("🖼️  Mostrando janela... (feche para continuar)")
            plt.show()
            
        except Exception as e:
            print(f"❌ Erro ao plotar série temporal: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_interactive(self):
        """Visualizador interativo com slider para navegar entre timestamps"""
        
        try:
            print("\n🎮 Iniciando modo interativo...")
            
            fig = plt.figure(figsize=(16, 7))
            
            # Criar subplots
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
            
            # Ajustar espaço para o slider
            plt.subplots_adjust(bottom=0.15)
            
            # Inicializar com primeiro timestamp
            ndvi = self.ds.ndvi.isel(time=0).values
            evi = self.ds.evi.isel(time=0).values
            timestamp = self.ds.time.values[0]
            date_str = np.datetime_as_string(timestamp, unit='D')
            
            # Plots iniciais
            im1 = ax1.imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1.0, interpolation='nearest')
            ax1.set_title(f'NDVI - {date_str}', fontsize=14, fontweight='bold')
            ax1.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('NDVI', fontsize=12)
            
            im2 = ax2.imshow(evi, cmap='RdYlGn', vmin=-0.2, vmax=2.0, interpolation='nearest')
            ax2.set_title(f'EVI - {date_str}', fontsize=14, fontweight='bold')
            ax2.axis('off')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('EVI', fontsize=12)
            
            # Criar slider
            ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
            slider = Slider(
                ax_slider, 
                'Timestamp', 
                0, 
                len(self.ds.time) - 1,
                valinit=0,
                valstep=1
            )
            
            # Função de atualização
            def update(val):
                time_idx = int(slider.val)
                ndvi = self.ds.ndvi.isel(time=time_idx).values
                evi = self.ds.evi.isel(time=time_idx).values
                timestamp = self.ds.time.values[time_idx]
                date_str = np.datetime_as_string(timestamp, unit='D')
                
                im1.set_data(ndvi)
                im2.set_data(evi)
                
                ax1.set_title(f'NDVI - {date_str}', fontsize=14, fontweight='bold')
                ax2.set_title(f'EVI - {date_str}', fontsize=14, fontweight='bold')
                
                fig.canvas.draw_idle()
            
            slider.on_changed(update)
            
            plt.suptitle(f'Navegador de Timestamps (Total: {len(self.ds.time)})', 
                         fontsize=16, fontweight='bold', y=0.98)
            
            print("\n💡 Use o slider para navegar entre os timestamps!")
            print("💡 Feche a janela para continuar...")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Erro no modo interativo: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_temporal_average(self):
        """Plota média temporal de NDVI e EVI"""
        
        try:
            print("\n📊 Calculando média temporal...")
            
            ndvi_mean = self.ds.ndvi.mean(dim='time').values
            evi_mean = self.ds.evi.mean(dim='time').values
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # NDVI médio
            im1 = ax1.imshow(ndvi_mean, cmap='RdYlGn', vmin=-0.2, vmax=1.0, interpolation='nearest')
            ax1.set_title('NDVI - Média Temporal', fontsize=14, fontweight='bold')
            ax1.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('NDVI Médio', fontsize=12)
            
            # EVI médio
            im2 = ax2.imshow(evi_mean, cmap='RdYlGn', vmin=-0.2, vmax=2.0, interpolation='nearest')
            ax2.set_title('EVI - Média Temporal', fontsize=14, fontweight='bold')
            ax2.axis('off')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('EVI Médio', fontsize=12)
            
            plt.suptitle(f'Média de {len(self.ds.time)} timestamps', 
                         fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            print("🖼️  Mostrando janela... (feche para continuar)")
            plt.show()
            
        except Exception as e:
            print(f"❌ Erro ao plotar média: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_std_deviation(self):
        """Plota desvio padrão temporal (variabilidade)"""
        
        try:
            print("\n📊 Calculando desvio padrão temporal...")
            
            ndvi_std = self.ds.ndvi.std(dim='time').values
            evi_std = self.ds.evi.std(dim='time').values
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # NDVI std
            im1 = ax1.imshow(ndvi_std, cmap='YlOrRd', vmin=0, vmax=0.3, interpolation='nearest')
            ax1.set_title('NDVI - Desvio Padrão Temporal', fontsize=14, fontweight='bold')
            ax1.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Variabilidade', fontsize=12)
            
            # EVI std
            im2 = ax2.imshow(evi_std, cmap='YlOrRd', vmin=0, vmax=0.5, interpolation='nearest')
            ax2.set_title('EVI - Desvio Padrão Temporal', fontsize=14, fontweight='bold')
            ax2.axis('off')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Variabilidade', fontsize=12)
            
            plt.suptitle('Variabilidade Temporal (maior = mais mudança)', 
                         fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            print("🖼️  Mostrando janela... (feche para continuar)")
            plt.show()
            
        except Exception as e:
            print(f"❌ Erro ao plotar desvio padrão: {e}")
            import traceback
            traceback.print_exc()
    
    def export_all_frames(self):
        """Exporta todos os timestamps como imagens"""
        output_dir = self.netcdf_path.parent / "frames"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n📸 Exportando {len(self.ds.time)} frames para {output_dir}...")
        
        for i in range(len(self.ds.time)):
            self.plot_single_timestamp(i, save=True)
            if (i + 1) % 5 == 0:
                print(f"   Processados {i + 1}/{len(self.ds.time)} frames...")
        
        print(f"✅ Todos os frames exportados para {output_dir}")
    
    def create_gif(self, duration: int = 500):
        """Cria GIF animado da série temporal"""
        try:
            from PIL import Image
            import io
            
            output_path = self.netcdf_path.parent / "animation.gif"
            frames = []
            
            print(f"\n🎬 Criando GIF animado...")
            
            for i in range(len(self.ds.time)):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                timestamp = self.ds.time.values[i]
                date_str = np.datetime_as_string(timestamp, unit='D')
                
                ndvi = self.ds.ndvi.isel(time=i).values
                evi = self.ds.evi.isel(time=i).values
                
                ax1.imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
                ax1.set_title(f'NDVI - {date_str}')
                ax1.axis('off')
                
                ax2.imshow(evi, cmap='RdYlGn', vmin=-0.2, vmax=2.0)
                ax2.set_title(f'EVI - {date_str}')
                ax2.axis('off')
                
                plt.tight_layout()
                
                # Converter para imagem
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()
                plt.close()
                
                if (i + 1) % 5 == 0:
                    print(f"   Processados {i + 1}/{len(self.ds.time)} frames...")
            
            # Salvar GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
            
            print(f"✅ GIF criado: {output_path}")
            
        except ImportError:
            print("❌ Pillow não instalado. Execute: pip install Pillow")
        except Exception as e:
            print(f"❌ Erro ao criar GIF: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="🔍 Visualizador de datasets HLS (NDVI/EVI)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "netcdf",
        type=str,
        help="Caminho para o arquivo NetCDF"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=['interactive', 'single', 'timeseries', 'mean', 'std', 'export', 'gif'],
        default='interactive',
        help="Modo de visualização (padrão: interactive)"
    )
    
    parser.add_argument(
        "--time-idx",
        type=int,
        default=0,
        help="Índice do timestamp para modo 'single' (padrão: 0)"
    )
    
    parser.add_argument(
        "--pixel-y",
        type=int,
        help="Coordenada Y do pixel para série temporal"
    )
    
    parser.add_argument(
        "--pixel-x",
        type=int,
        help="Coordenada X do pixel para série temporal"
    )
    
    args = parser.parse_args()
    
    # Detectar backend disponível
    print(f"🔧 Backend matplotlib: {matplotlib.get_backend()}")
    
    # Criar visualizador
    viewer = DatasetViewer(args.netcdf)
    
    if not viewer.load_dataset():
        return 1
    
    # Executar modo selecionado
    try:
        if args.mode == 'interactive':
            viewer.plot_interactive()
        
        elif args.mode == 'single':
            viewer.plot_single_timestamp(args.time_idx)
        
        elif args.mode == 'timeseries':
            viewer.plot_time_series(args.pixel_y, args.pixel_x)
        
        elif args.mode == 'mean':
            viewer.plot_temporal_average()
        
        elif args.mode == 'std':
            viewer.plot_std_deviation()
        
        elif args.mode == 'export':
            viewer.export_all_frames()
        
        elif args.mode == 'gif':
            viewer.create_gif()
            
    except Exception as e:
        print(f"\n❌ Erro ao executar modo '{args.mode}': {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())